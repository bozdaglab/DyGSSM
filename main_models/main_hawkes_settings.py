import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
import sys
import os
from pathlib import Path
repo_path = str(Path(__file__).parent.parent / "hawkesGNN")
sys.path.append(repo_path)
from hawkesGNN.utils import seed_everything, generate_random_seeds, save_result
import hawkesGNN.models
sys.modules["models"] = hawkesGNN.models



def build_model(args, factory, device):
    from DyGSSM.DyGSSM import DyGSSM, HIPPO
    model = DyGSSM(n_noeds=factory.num_nodes, in_features=factory.node_feats_dim, hidden_dim=args.n_hidden, 
                        dropout=args.dropout, hidden=64, out_features=64, 
            num_layers=2, 
            num_heads=1, 
            device=device,
            fused_model="semantic",
            bidirectional=False,
            message_pass_type="GCN").to(device)
    fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
    state_dim = len(fast_weights)
    hipoo = HIPPO(state_dim=state_dim)
    return model, hipoo



def main_hawkes(args, path_1, model_type):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.set_default_dtype(torch.float64)

    print(args)
    seed_everything(args.seed)     # for init negative_sampling

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    import torch_geometric.transforms as T
    from hawkesGNN.datasets import BitcoinOTC, BitcoinAlpha, UCIMessage, AS733, SBM, StackOverflow, RedditTitle, RedditBody
    transform = T.Compose([T.RemoveDuplicatedEdges(reduce='max')]) # for usi-message

    DS = {'bitcoinotc': BitcoinOTC, 'bitcoinalpha': BitcoinAlpha, 'redt': RedditTitle, 'redb': RedditBody,
            'uci': UCIMessage, 'as733': AS733, 'as733_full': AS733, 'sbm': SBM, 'stackoverflow': StackOverflow}
    split = {
        'bitcoinotc': [95, 95+14, 95+14+28], 'bitcoinalpha': [95, 95+13, 95+13+28], 
        'uci': [35,40,50], 'as733': [70, 70+10, 70+10+20], 
        'uci': [61, 61+9, 61+9+17],
        'redt': [122, 122+35, 122+35+17],
        'redb': [122, 122+35, 122+35+17],
        'sbm': [35,40,50],
        'stackoverflow': [70, 70+10, 70+10+20],
        'as733_full': [int(733*0.7), int(733*0.8), int(733*1)], 
        'wikipedia': [0.7, 0.85, 1], 
        'mooc': [0.7, 0.85, 1], 
        'lastfm': [0.7, 0.85, 1], 
        'reddit': [0.7, 0.85, 1]
    }
    
    if args.test:
        #args.window=3
        for k in split.keys():
            split[k] = [10, 12, 14]
        split['uci'] = [21, 21 + 3, 21 + 3 + 6]
        #split['as733'] = [int(733*0.7), int (733*0.8), 733]

    root = {'bitcoinotc': './data/bitcoin', 'bitcoinalpha': './data/bitcoin', 'redt': './data/reddit', 'redb': './data/reddit',
        'uci': './data/uci-msg', 'as733': './data/as-733', 'sbm': './data/sbm', 'as733_full': './data/as-733',
        'stackoverflow': './data/stackoverflow'}
    
    if args.dataset in ['bitcoinotc', 'bitcoinalpha', 'uci', 'as733', 'as733_full', 'redt', 'redb', 'sbm', 'stackoverflow']:
        from hawkesGNN.dataloader import BitcoinLoaderFactory
        dataset = DS[args.dataset](root[args.dataset], transform=transform)
        factory = BitcoinLoaderFactory(dataset, 
            node_feat_type=args.node_feat, negative_sampling=args.n_neg_test)
    elif args.dataset in ['wikipedia', 'mooc', 'lastfm', 'reddit']:
        from torch_geometric.datasets import JODIEDataset
        from hawkesGNN.dataloader import JodieLoaderFactory
        import os
        args.n_neg_test = 1  # args.n_neg_test should == 1
        fo = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(fo, 'data', 'jodie')
        print(root)
        factory = JodieLoaderFactory(root, args.dataset, negative_sampling=args.n_neg_test) 
    

    loaders = factory.get_roland_snaps(split=split[args.dataset], device=device)


    from train_test_models.train_hawkes_settings import DyGSSMLinkPrediction
    lp = DyGSSMLinkPrediction(args, build_model)
    
    if args.dataset in ['wikipedia', 'mooc', 'lastfm', 'reddit']:
        min_dst_idx, max_dst_idx = int(factory.data.dst.min()), int(factory.data.dst.max())
        lp.set_jodie(min_dst_idx, max_dst_idx)
    lp.main(args,  factory,  loaders, device, path_1, model_type)
