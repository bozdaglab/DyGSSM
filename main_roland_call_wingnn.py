import os
os.environ['MPLCONFIGDIR'] = "/tmp"
import random
import numpy as np
import torch
import logging
import sys
sys.path.append("/nfs/home/UNT/ba0499/ba0499/Time_series_graph/roland-master")
from graphgym.cmd_args import parse_args
from graphgym.config import (cfg, assert_cfg, dump_cfg,
                             update_out_dir, get_parent_dir)
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import setup_printing, create_logger
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.model_builder import create_model
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.contrib.train import *
from graphgym.register import train_dict

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


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

    elif name in ['as-733.tar.gz']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
        cfg.train.start_compute_mrr = 81

    elif name in ['sx-stackoverflow.zip']:
        cfg.dataset.format = 'stackoverflow'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'M'
        # cfg.train.start_compute_mrr = 81

    else:
        raise ValueError(f'No default config for dataset {name}.')


def dataset_cfg_setup_live_update(name: str):
    """
    Setup required fields in cfg for the given dataset.
    """
    if name in ['bsi_svt_2008.tsv', 'bsi_zk_2008.tsv']:
        cfg.dataset.format = 'roland_bsi_general'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland'
        cfg.dataset.node_encoder = True
        if name == 'bsi_svt_2008.tsv':
            cfg.transaction.feature_node_int_num = [1018, 33, 13, 23, 5]
            cfg.transaction.snapshot_freq = 'W'
        else:
            cfg.transaction.feature_node_int_num = [1483, 32, 13, 24, 5]
            cfg.transaction.snapshot_freq = 'D'

    elif name in ['bitcoinotc.csv', 'bitcoinalpha.csv']:
        cfg.dataset.format = 'bitcoin'
        cfg.dataset.edge_dim = 2
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['CollegeMsg.txt']:
        cfg.dataset.format = 'uci_message'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['reddit-body.tsv', 'reddit-title.tsv']:
        cfg.dataset.format = 'reddit_hyperlink'
        cfg.dataset.edge_dim = 88
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'W'

    elif name in ['AS-733']:
        cfg.dataset.format = 'as'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'D'
    elif name in ['sx-stackoverflow']:
        cfg.dataset.format = 'stackoverflow'
        cfg.dataset.edge_dim = 1
        cfg.dataset.edge_encoder_name = 'roland_general'
        cfg.dataset.node_encoder = False
        cfg.transaction.snapshot_freq = 'M'

    else:
        raise ValueError(f'No default config for dataset {name}.')


def call(i, args):
    # Load cmd line args
    # args = parse_args()
    # Repeat for different random seeds
    # for i in range(args.repeat):
    # Load config file
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)

    # over-ride data path and remark if required.
    if args.override_data_dir is not None:
        cfg.dataset.dir = args.override_data_dir
    if args.override_remark is not None:
        cfg.remark = args.override_remark

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    cfg.seed = i + 1
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    update_out_dir(out_dir_parent, args.cfg_file)
    dump_cfg(cfg)
    setup_printing()
    auto_select_device()
    cfg.train.mode = 'baseline_v2'
    # Infer dataset format if required, retrieve the cfg setting associated
    # with the particular dataset.
    if cfg.dataset.format == 'infer':
        if cfg.train.mode in ['baseline', 'baseline_v2', 'live_update_fixed_split']:
            dataset_cfg_setup_fixed_split(cfg.dataset.name)
        elif cfg.train.mode == 'live_update':
            dataset_cfg_setup_live_update(cfg.dataset.name)
        elif cfg.train.mode == 'live_update_baseline':
            dataset_cfg_setup_live_update(cfg.dataset.name)

    # Set learning environment
    if cfg.dataset.premade_datasets in ['fresh', 'fresh_save_cache']:
        datasets = create_dataset()
    else:
        datasets = torch.load(os.path.join(cfg.dataset.dir,
                                            cfg.dataset.premade_datasets))
    return datasets, cfg.dataset.name.split(".")[0]
    
