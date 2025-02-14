import torch
import dgl
import random
from torch_geometric.index import index2ptr
from torch_geometric.utils import sort_edge_index
from torch import Tensor
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
from collections import Counter
import pickle
from dask_ml.preprocessing import OrdinalEncoder
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from run.main_roland_call_wingnn import call
from dataset_prep import load_r
import os

def load_r_custom_dblp(e_feat, n_feat_, csv_file, ts_):

    unique_subreddits = pd.unique(
        csv_file[['src', 'dst']].to_numpy().ravel())
    unique_subreddits = np.sort(unique_subreddits)
    cate_type = pd.api.types.CategoricalDtype(categories=unique_subreddits,
                                              ordered=True)
    csv_file['src'] = csv_file['src'].astype(
        cate_type)
    csv_file['dst'] = csv_file['dst'].astype(
        cate_type)
    enc = OrdinalEncoder(columns=['src', 'dst'])
    df_encoded = enc.fit_transform(csv_file)
    df_encoded.reset_index(drop=True, inplace=True)
    for col in ['src', 'dst']:
        assert all(unique_subreddits[df_encoded[col]] == csv_file[col])
    num_nodes = len(cate_type.categories)
    node_feature = torch.ones(size=(num_nodes, n_feat_.shape[1]))
    node_feature = node_feature * np.mean(csv_file.values)
    for i, subreddit in enumerate(cate_type.categories):
        if subreddit in pd.DataFrame(n_feat_).index:
            embedding = pd.DataFrame(n_feat_).loc[subreddit]
            node_feature[i, :] = torch.Tensor(embedding.values)
    edge_feature = torch.Tensor(e_feat).float()
    edge_index = torch.Tensor(
        df_encoded[['src',
                    'dst']].values.transpose()).long()
    num_nodes = torch.max(edge_index) + 1

    start_date = '2021-01-01'
    date_range = pd.date_range(start=start_date, periods=27, freq='M')

    seconds_since_epoch = (date_range - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')


    for t, time in zip(ts_, seconds_since_epoch.tolist()):
        df_encoded['ts'] = df_encoded['ts'].replace(t, time)
    edge_time = torch.FloatTensor(df_encoded['ts'].values)
    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    t = graph.edge_time.numpy().astype(np.int64)
    snapshot_freq = "M"
    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(graph.edge_time)))
    freq_map = {'D': '%j',
                'W': '%W',
                'M': '%m'
                }
    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)
    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)
    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    periods = sorted(list(period2id.keys()))
    snapshot_list = list()
    for p in periods:
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=graph.node_feature,
            edge_feature=graph.edge_feature[period_members, :],
            edge_index=graph.edge_index[:, period_members],
            edge_time=graph.edge_time[period_members],
            directed=graph.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list

def make_usci_data(graphs, n_feat, e_feat, e_time, n_node, n_dim, device):
    graph_l = []
    for idx, graph in tqdm(enumerate(graphs)):
        graph_d = dgl.from_scipy(graph)
        graph_d.edge_feature = torch.Tensor(e_feat[idx])
        graph_d.edge_time = torch.Tensor(e_time[idx])
        if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
            n_feat_t = graph_l[idx - 1].node_feature
            graph_d.node_feature = torch.Tensor(n_feat_t)
        else:
            graph_d.node_feature = torch.Tensor(n_feat[idx])
        graph_d = dgl.remove_self_loop(graph_d)
        graph_d = dgl.add_self_loop(graph_d)
        edges = graph_d.edges()
        row = edges[0].numpy()
        col = edges[1].numpy()
        n_e = graph_d.num_edges() - graph_d.num_nodes()
        y_pos = np.ones(shape=(n_e,))
        y_neg = np.zeros(shape=(n_e,))
        y = list(y_pos) + list(y_neg)
        edge_label_index = list()
        edge_label_index.append(row.tolist()[:n_e])
        edge_label_index.append(col.tolist()[:n_e])
        graph_d.edge_label = torch.Tensor(y)
        graph_d.edge_label_index = torch.LongTensor(edge_label_index)
        graph_d.random_walk_node2vec_f = generate_walks(graph_d, 5)
        graph_l.append(graph_d.to(device))

    for idx, graph in tqdm(enumerate(graphs)):
        graph = Graph(
            node_feature=graph_l[idx].node_feature,
            edge_feature=graph_l[idx].edge_feature,
            edge_index=graph_l[idx].edge_label_index,
            edge_time=graph_l[idx].edge_time,
            directed=True
        )

        dataset = GraphDataset(graph,
                            task='link_pred',
                            edge_negative_sampling_ratio=1.0,
                            minimum_node_per_graph=5)
        edge_labe_index = dataset.graphs[0].edge_label_index
        graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)
    return graph_l

def make_data_deepsnap(datasets, dataset_name, idx_graph, path, device):
    graph_l = []
    for data in tqdm(datasets):
        # graph_l = dgl.DGLHeteroGraph()
        graph_d = dgl.from_scipy(data)
        graph_d.edge_feature =  data.edge_feature
        graph_d.edge_time = data.edge_time
        graph_d.node_feature = data.node_feature
        graph_d.edge_index = data.edge_index
        graph_d = dgl.remove_self_loop(graph_d)
        graph_d = dgl.add_self_loop(graph_d)
        edges = graph_d.edges()
        graph_d.edge_label_index = data.edge_label_index
        graph_d.edge_label = data.edge_label
        row = edges[0].numpy()
        col = edges[1].numpy()
        n_e = graph_d.num_edges() - graph_d.num_nodes()
        y_pos = np.ones(shape=(n_e,))
        y_neg = np.zeros(shape=(n_e,))
        y = list(y_pos) + list(y_neg)
        edge_label_index = list()
        edge_label_index.append(row.tolist()[:n_e])
        edge_label_index.append(col.tolist()[:n_e])
        graph_d.edge_label = torch.Tensor(y)
        graph_d.edge_label_index = torch.LongTensor(edge_label_index)
        graph_d.random_walk_node2vec_f = generate_walks(graph_d, 5)
        # graph_d.random_walk_node2vec_f = random_walk_2(graph_d, node2vec=True)
        graph_l.append(graph_d)
    # Negative sample sampling 1:1
    for idx, graph in enumerate(datasets):
        graph = Graph(
            node_feature=graph_l[idx].node_feature,
            edge_feature=graph_l[idx].edge_feature,
            edge_index=graph_l[idx].edge_label_index,
            edge_time=graph_l[idx].edge_time,
            directed=True,
        )

        dataset = GraphDataset(
            graph,
            task="link_pred",
            edge_negative_sampling_ratio=1.0,
            minimum_node_per_graph=5,
        )
        edge_labe_index = dataset.graphs[0].edge_label_index
        graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)

    with open(f"{path}/{dataset_name}_{idx_graph}.pkl", "wb") as file:
        pickle.dump(graph_l, file)
    return graph_l


def load_data(path, rep, args,path_1, device):
    dataset_name = args.dataset
    # if os.path.exists(f"{path}/{dataset_name}_{rep}.pkl"):
        # with open(f"{path}/{dataset_name}_{rep}.pkl", "rb") as file:
        #     graph_l = pickle.load(file)
    # else:
    if args.dataset == 'dblp':
        e_feat = np.load(f'{path_1}/dataset/{dataset_name}/ml_{dataset_name}.npy')
        n_feat_ = np.load(f'{path_1}/dataset/{dataset_name}/ml_{dataset_name}_node.npy')
        ts_ = np.load(f'{path_1}/dataset/{dataset_name}/ml_{dataset_name}_ts.npy')
        csv_file = pd.read_csv(f'{path_1}/dataset/{dataset_name}/ml_{dataset_name}.csv')
        datasets = load_r_custom_dblp(e_feat, n_feat_, csv_file, ts_)
        
        graph_l = make_data_deepsnap(datasets, dataset_name, rep, path, device)
    elif args.dataset in ["reddit_body", "reddit-title", 
                        "bitcoinotc", "bitcoinalpha",
                        'stackoverflow_M']:
        datasets, dataset_name = call(rep, args)
        graph_l = make_data_deepsnap(datasets, dataset_name, rep, path, device)
    elif args.dataset == "uci-msg":
        graphs, e_feat, e_time, n_feat = load_r(args.dataset)
        n_dim = n_feat[0].shape[1]
        n_node = n_feat[0].shape[0]
        graph_l = make_usci_data(graphs, n_feat, e_feat, e_time, n_node, n_dim, device)
    else:
        raise ValueError
    # with open(f"{path}/{dataset_name}_{rep}.pkl", "wb") as file:
    #     pickle.dump(graph_l, file)
    return graph_l


def generate_walks(graph, num_walks):
    final_walks = np.zeros((graph.num_nodes(), num_walks))
    epochs = 30
    num_edges = num_walks - 1
    nodes = graph.nodes().repeat_interleave(epochs)
    sequence = dgl.sampling.node2vec_random_walk(graph, nodes, 0.5, 2.0, num_edges)
    for track, idx in tqdm(enumerate(range(0, len(sequence), epochs))):
        most_freq = Counter(sequence[idx: idx + epochs].view(1, -1)[0].tolist()).most_common(num_walks + 1)
        walks = find_most_frequent(most_freq, track, num_walks, num_walks - 1)
        final_walks[track] = np.concatenate([[track], walks])
    return torch.tensor(final_walks.astype(int))


def find_most_frequent(most_freq, idx, walk_len, walk_minus):
    if len(most_freq) >= walk_len: 
        walks_ = [
        node_num 
        for node_num, _ in most_freq 
        if not node_num == idx
        ][:walk_minus]
    else:
        walks_ = [node_num for node_num, _ in most_freq]
        walks_new = [
        node_num 
        for node_num in walks_ 
        if not node_num == idx
        ]
        if len(walks_new) == 0:
            walks_ = np.repeat(walks_, walk_minus).tolist()
        elif len(walks_new) < walk_minus:
            number_to_pick = walk_minus - len(walks_new)
            walks_.remove(idx)
            for _ in range(number_to_pick):
                walks_new.extend(random.sample(walks_, k=1))
            walks_ = walks_new

    return walks_
