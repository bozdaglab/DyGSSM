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
# sys.path.append(Path(__file__).parent / "roland-master")
from roland.run.main_roland_call_wingnn import call
from pre_processing.dataset_prep import load_r
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

def check_rw_needed(prev, row, col):
    try:
        set1 = set(map(tuple, torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).t().tolist()))
    except TypeError:
        set1 = set(map(tuple, torch.stack([row, col]).t().tolist()))
    set2 = set(map(tuple, prev.tolist()))
    effected_edges = set1 - set2
    return torch.tensor(list(effected_edges)).t(), True if len(effected_edges)> 0 else False



def make_usci_data(graphs, 
                   device,
                   n_feat=None, 
                   e_feat=None, 
                   e_time=None, 
                   n_node=None, 
                   n_dim=None,
                    hawkes=False):
    graph_l = []
    prev = None
    for idx, graph in tqdm(enumerate(graphs)):
        if hawkes:
            graph_d = dgl.from_scipy(Graph(
                node_feature=graph.x,
                edge_index=graph.edge_index,
                directed=True
            ))
            graph_d.node_feature = graph.x
            graph_d.edge_index = graph.edge_index
        else:
            graph_d = dgl.from_scipy(graph)
            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
                n_feat_t = graph_l[idx - 1].node_feature
                graph_d.node_feature = torch.Tensor(n_feat_t)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])
            graph_d.edge_feature = torch.Tensor(e_feat[idx])
        

        graph_d = dgl.remove_self_loop(graph_d)
        graph_d = dgl.add_self_loop(graph_d)
        edges = graph_d.edges()
        row = edges[0]
        col = edges[1]
        n_e = graph_d.num_edges() - graph_d.num_nodes()
        y_pos = np.ones(shape=(n_e,))
        y_neg = np.zeros(shape=(n_e,))
        y = list(y_pos) + list(y_neg)
        edge_label_index = list()
        edge_label_index.append(row.numpy().tolist()[:n_e])
        edge_label_index.append(col.numpy().tolist()[:n_e])
        graph_d.edge_label = torch.Tensor(y)
        graph_d.edge_label_index = torch.LongTensor(edge_label_index)

        if prev is None:
            graph_d.random_walk_node2vec_f, _ = random_walk_all(graph, graph_d.num_nodes(), walk_length=5, n_runs=30, top_k=4, seed=42)
        # graph_d.random_walk_node2vec_f, _ = generate_walks(graph=graph_d, num_walks=5)
        else:
            effected_nodes, need_rw = check_rw_needed(prev, row, col)
            if need_rw:
                current_walk, masks = random_walk_all(graph, graph_d.num_nodes(), walk_length=5, n_runs=30, top_k=4, seed=42, effected_nodes=effected_nodes)
                # generate_walks(graph=graph_d, effected_nodes=effected_nodes, num_walks=5)
                current_walk[torch.where(~masks)[0]] = graph_l[idx - 1].random_walk_node2vec_f[torch.where(~masks)[0]]
                graph_d.random_walk_node2vec_f = current_walk
            else:
                graph_d.random_walk_node2vec_f = graph_d[idx - 1].random_walk_node2vec_f
        graph_l.append(graph_d.to(device))
        prev = torch.stack([row, col]).t()
    if hawkes:
        return graph_l
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
    prev = None
    
    for idx, data in tqdm(enumerate(datasets)):
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
        if prev is None:
            graph_d.random_walk_node2vec_f, _ = generate_walks(graph=graph_d, num_walks=5)
        else:
            effected_nodes, need_rw = check_rw_needed(prev, row, col)
            if need_rw:
                current_walk, masks = generate_walks(graph=graph_d, effected_nodes=effected_nodes, num_walks=5)
                current_walk[torch.where(~masks)[0]] = graph_l[idx - 1].random_walk_node2vec_f[torch.where(~masks)[0]]
                graph_d.random_walk_node2vec_f = current_walk
            else:
                graph_d.random_walk_node2vec_f = graph_d[idx - 1].random_walk_node2vec_f

        # graph_d.random_walk_node2vec_f = random_walk_2(graph_d, node2vec=True)
        graph_l.append(graph_d.to(device))
        try:
            prev = torch.stack([row, col]).t()
        except TypeError:
            prev = torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).t()
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

    return graph_l


def load_data(rep, args, path, path_1, dataset_name, device, model_type):
    if os.path.exists(f"{path_1}/processed_data_{model_type}/{dataset_name}/{rep}.pkl"):
        with open(f"{path_1}/processed_data_{model_type}/{dataset_name}/{rep}.pkl", "rb") as file:
            graph_l = pickle.load(file)
    else:
        if dataset_name == 'dblp':
            e_feat = np.load(f'{path}/dataset/{dataset_name}/ml_{dataset_name}.npy')
            n_feat_ = np.load(f'{path}/dataset/{dataset_name}/ml_{dataset_name}_node.npy')
            ts_ = np.load(f'{path}/dataset/{dataset_name}/ml_{dataset_name}_ts.npy')
            csv_file = pd.read_csv(f'{path}/dataset/{dataset_name}/ml_{dataset_name}.csv')
            datasets = load_r_custom_dblp(e_feat, n_feat_, csv_file, ts_)
            
            graph_l = make_data_deepsnap(datasets, dataset_name, rep, path_1, device)
        elif dataset_name in ["reddit-title", "bitcoinotc", "bitcoinalpha"]:
            datasets, _ = call(rep, args)
            graph_l = make_data_deepsnap(datasets, dataset_name, rep, path, device)
        elif dataset_name == "uci-msg":
            graphs, e_feat, e_time, n_feat = load_r(dataset_name, path)
            n_dim = n_feat[0].shape[1]
            n_node = n_feat[0].shape[0]
            graph_l = make_usci_data(graphs=graphs, 
                                     n_feat=n_feat, 
                                     e_feat=e_feat, 
                                     e_time=e_time, 
                                     n_node=n_node, 
                                     n_dim=n_dim, 
                                     device=device)
        else:
            raise ValueError
        with open(f"{path_1}/processed_data_{model_type}/{dataset_name}/{rep}.pkl", "wb") as file:
            pickle.dump(graph_l, file)

    return graph_l


import scipy.sparse as sp
def random_walk_all(adj_csr, num_nodes, walk_length=5, n_runs=30, top_k=4, seed=42, effected_nodes=None):
    adj_csr = adj_csr + adj_csr.T
    adj_csr = sp.csr_matrix(adj_csr)
    rng = np.random.default_rng(seed)
    final_walks = np.zeros((num_nodes, 5))
    if effected_nodes is None:
        mask = None
    else:
        all_nodes = torch.tensor(list(set(torch.cat([effected_nodes[0], effected_nodes[1]]).tolist())))
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[all_nodes] = True
    # all_frequent_nodes = {}
    n_nodes = adj_csr.shape[0]
    
    for node in range(n_nodes):
        all_walks = []
        for _ in range(n_runs):
            walk = [node]
            current = node
            for _ in range(walk_length-1):
                neighbors = adj_csr.indices[adj_csr.indptr[current]:adj_csr.indptr[current+1]]
                if len(neighbors) == 0:
                    break
                current = rng.choice(neighbors)
                walk.append(current)
            all_walks.extend(walk)
        
        # count most frequent nodes visited (excluding the source itself)
        counts = Counter(all_walks)
        counts.pop(node, None)
        most_common = [n for n, _ in counts.most_common(top_k)]
        
        # --- pad to length 5 ---
        if len(most_common) == 0:
            # if no sequence, repeat the source node 5 times
            padded = [node] * top_k
        elif len(most_common) < top_k:
            # pad randomly with other nodes
            needed = top_k - len(most_common)
            candidates = list(set(range(n_nodes)) - set(most_common) - {node})
            if len(candidates) >= needed:
                extra = rng.choice(candidates, size=needed, replace=False).tolist()
            else:
                # if not enough candidates, allow repeats
                extra = rng.choice(n_nodes, size=needed, replace=True).tolist()
            padded = most_common + extra
        else:
            padded = most_common[:top_k]
        final_walks[node] = np.concatenate([[node], padded])
        # all_frequent_nodes[node] = padded
    
    return torch.tensor(final_walks.astype(int)), mask

def generate_walks(graph, num_walks, effected_nodes=None):
    epochs = 30
    num_edges = num_walks - 1
    final_walks = np.zeros((graph.num_nodes(), num_walks))
    if effected_nodes is None:
        nodes = graph.nodes().repeat_interleave(epochs)
        mask = None
    else:
        all_nodes = torch.tensor(list(set(torch.cat([effected_nodes[0], effected_nodes[1]]).tolist())))
        nodes = all_nodes.repeat_interleave(epochs)
        mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        mask[all_nodes] = True
    sequence = dgl.sampling.node2vec_random_walk(graph, nodes, 0.5, 2.0, num_edges)
    for idx in tqdm(range(0, len(sequence), epochs)):
        seq = sequence[idx: idx + epochs]
        most_freq = Counter(seq.view(1, -1)[0].tolist()).most_common(num_walks + 1)
        track = sequence[idx: idx + epochs][0][0].item()
        walks = find_most_frequent(most_freq, track, num_walks, num_walks - 1)
        final_walks[track] = np.concatenate([[track], walks])
    return torch.tensor(final_walks.astype(int)), mask


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
            try:
                walks_.remove(idx)
            except ValueError:
                pass
            for _ in range(number_to_pick):
                walks_new.extend(random.sample(walks_, k=1))
            walks_ = walks_new

    return walks_
