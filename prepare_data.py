
from graphgym.config import cfg
import networkx as nx
import time
import logging
import pickle

from deepsnap.dataset import GraphDataset
import torch
from torch.utils.data import DataLoader

from torch_geometric.datasets import *
import torch_geometric.transforms as T
from graphgym.config import cfg
import graphgym.models.feature_augment as preprocess
from graphgym.models.transform import (ego_nets, remove_node_feature,
                                       edge_nets, path_len)
from graphgym.contrib.loader import *
import graphgym.register as register

from ogb.graphproppred import PygGraphPropPredDataset
from deepsnap.batch import Batch

import pdb


def dataset_cfg_setup_fixed_split(name: str):
    """
    Setup required fields in cfg for the given dataset.
    """
    if name in ['bsi_svt_2008.tsv', 'bsi_zk_2008.tsv']:
        cfg.dataset.format = 'roland_bsi_general'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland'
        cfg.dataset.node_encoder = True
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = 'D'

    elif name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
        cfg.dataset.format = 'bitcoin'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = '1200000s'
        if name == 'bitcoinotc.csv':
            cfg.train.start_compute_mrr = 110
        else:
            cfg.train.start_compute_mrr = 109

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.71, 0.09, 0.2]
        cfg.transaction.snapshot_freq = '190080s'
        cfg.train.start_compute_mrr = 72

    elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.dataset.split = [0.7, 0.1, 0.2]
        cfg.transaction.snapshot_freq = 'D'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
        cfg.train.start_compute_mrr = 81

    else:
        raise ValueError(f'No default config for dataset {name}.')




def load_pyg(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset_raw = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset_raw = TUDataset(dataset_dir, name,
                                    transform=T.Constant())
        else:
            dataset_raw = TUDataset(dataset_dir, name[3:])
        # TU_dataset only has graph-level label
        # The goal is to have synthetic tasks
        # that select smallest 100 graphs that have more than 200 edges
        if cfg.dataset.tu_simple and cfg.dataset.task != 'graph':
            size = []
            for data in dataset_raw:
                edge_num = data.edge_index.shape[1]
                edge_num = 9999 if edge_num < 200 else edge_num
                size.append(edge_num)
            size = torch.tensor(size)
            order = torch.argsort(size)[:100]
            dataset_raw = dataset_raw[order]
    elif name == 'Karate':
        dataset_raw = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset_raw = Coauthor(dataset_dir, name='CS')
        else:
            dataset_raw = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset_raw = Amazon(dataset_dir, name='Computers')
        else:
            dataset_raw = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset_raw = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset_raw = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset_raw = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))
    graphs = GraphDataset.pyg_to_graphs(dataset_raw)
    return graphs


def load_nx(name, dataset_dir):
    '''
    load networkx format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    try:
        with open('{}/{}.pkl'.format(dataset_dir, name), 'rb') as file:
            graphs = pickle.load(file)
    except:
        graphs = nx.read_gpickle('{}.gpickle'.format(dataset_dir, name))
        if not isinstance(graphs, list):
            graphs = [graphs]
    return graphs


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    cfg.dataset.dir = '/bozdagpool/UNT/ba0499/Codes/WinGNN-main/dataset/roland_public_data'
    format = cfg.dataset.format
    name = cfg.dataset.name
    # dataset_dir = '{}/{}'.format(cfg.dataset.dir, name)
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        graphs = func(format, name, dataset_dir)
        if graphs is not None:
            return graphs
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        graphs = load_pyg(name, dataset_dir)
    # Load from networkx formatted data
    # todo: clean nx dataloader
    elif format == 'nx':
        graphs = load_nx(name, dataset_dir)
    # Load from OGB formatted data
    elif cfg.dataset.format == 'OGB':
        if cfg.dataset.name == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
            graphs = GraphDataset.pyg_to_graphs(dataset)
        # Note this is only used for custom splits from OGB
        split_idx = dataset.get_idx_split()
        return graphs, split_idx
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    return graphs


def filter_graphs():
    '''
    Filter graphs by the min number of nodes
    :return: min number of nodes
    '''
    if cfg.dataset.task == 'graph':
        min_node = 0
    else:
        min_node = 5
    return min_node



def create_dataset():
    ## Load dataset
    time1 = time.time()
    if cfg.dataset.format in ['OGB']:
        graphs, splits = load_dataset()
    else:
        graphs = load_dataset()

    ## Filter graphs
    time2 = time.time()
    min_node = filter_graphs()

    ## Create whole dataset
    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        minimum_node_per_graph=min_node)

    dataset


def create_loader(datasets):
    loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)

    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False))

    return loaders