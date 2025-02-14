#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :load datasets

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from deepsnap.graph import Graph

def load(nodes_num):
    """
    load_dataset
    :param nodes_num:
    :return:
    """
    path = "dataset/dblp_timestamp/"

    train_e_feat_path = path + 'train_e_feat/' + type + '/'
    test_e_feat_path = path + 'test_e_feat/' + type + '/'

    train_n_feat_path = path + type + '/' + 'train_n_feat/'
    test_n_feat_path = path + type + '/' + 'test_n_feat/'


    path = path + type
    train_path = path + '/train/'
    test_path = path + '/test/'

    train_n_feat = read_e_feat(train_n_feat_path)
    test_n_feat = read_e_feat(test_n_feat_path)

    train_e_feat = read_e_feat(train_e_feat_path)
    test_e_feat = read_e_feat(test_e_feat_path)

    num = 0
    train_graph = read_graph(train_path, nodes_num, num)
    num = num + len(train_graph)
    test_graph = read_graph(test_path, nodes_num, num)
    return train_graph, train_e_feat, train_n_feat, test_graph, test_e_feat, test_n_feat


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
    from dask_ml.preprocessing import OrdinalEncoder
    import torch
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
                    'dst']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1

    start_date = '2021-01-01'  # Change this to your desired starting date

    # Generate a range of 27 months
    date_range = pd.date_range(start=start_date, periods=27, freq='M')

    # Convert to seconds since epoch
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
    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }
    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)
    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)
    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    periods = sorted(list(period2id.keys()))
    snapshot_list = list()
    for p in periods:
        # unique IDs of edges in this period.
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


def load_r(name):
    path = Path(__file__).parent
    path = f"{path}/" + "dataset/" + name
    path_ei = path + '/' + 'edge_index/'
    path_nf = path + '/' + 'node_feature/'
    path_ef = path + '/' + 'edge_feature/'
    path_et = path + '/' + 'edge_time/'

    edge_index = read_npz(path_ei)
    edge_feature = read_npz(path_ef)
    node_feature = read_npz(path_nf)
    edge_time = read_npz(path_et)

    nodes_num = node_feature[0].shape[0]

    sub_graph = []
    for e_i in edge_index:
        row = e_i[0]
        col = e_i[1]
        ts = [1] * len(row)
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph, edge_feature, edge_time, node_feature


def read_npz(path):
    filesname = os.listdir(path)
    npz = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('.')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        npz.append(np.load(path+filename))

    return npz


def read_e_feat(path):
    filesname = os.listdir(path)
    e_feat = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        e_feat.append(np.load(path+filename))

    return e_feat


def read_graph(path, nodes_num, num):

    filesname = os.listdir(path)
    # 对文件名做一个排序
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id) - num
        file_s[id] = filename

    # 文件读取
    sub_graph = []
    for file in file_s:
        sub_ = pd.read_csv(path + file)

        row = sub_.src_l.values
        col = sub_.dst_l.values

        node_m = set(row).union(set(col))
        # ts = torch.Tensor(sub_.timestamp.values)
        ts = [1] * len(row)

        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph

