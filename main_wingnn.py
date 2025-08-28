import pandas as pd
import math
import torch
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import models
from test import test
from train import train
from config import cfg
from Logger import getLogger
from utils_helper import create_optimizer
from helper import load_data
from get_args import load_args
import warnings
from itertools import product
import os
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    path_1 = str(Path(__file__).parent)
    args = load_args()
    if not os.path.exists(f"{path_1}/results/{args.dataset}"):
        os.makedirs(f"{path_1}/results/{args.dataset}")
        os.makedirs(f"{path_1}/processed_data/{args.dataset}")
        os.makedirs(f"{path_1}/best_models/{args.dataset}")
        
    hyperparameters  = {
        'lr' : [0.007, 0.003],
        'weight_decay' : [0.0001],
        'message_pass_type': ["GCN"],
    }
    combinations = list(product(*hyperparameters.values()))
    logger = getLogger(cfg.log_path)
    
    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = 0
    final_result = defaultdict(list)
    for combination in combinations:
        hyper = {
        'lr' :combination[0],
        'weight_decay' :combination[1],
        'message_pass_type': combination[2],
    }
        dic_keys = '_'.join([str(i) for i in combination])
        for rep in range(0, args.repeat):
            path = str(Path(__file__).parent)
            dataset_name = args.dataset
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            graph_l = load_data(path, rep, args, path_1, device)

            n_dim = graph_l[0].node_feature.shape[1]
            n_node = graph_l[0].num_nodes()

            
            model = models.DyGSSM(
                n_noeds=n_node,
                in_features=n_dim,
                out_features=args.out_dim, 
                hidden_dim=args.num_hidden, 
                hidden=args.hidden_dim, 
                num_layers=args.num_layers, 
                dropout=args.dropout,
                num_heads=1, 
                device=device,
                fused_model="semantic",
                bidirectional=False,
                message_pass_type=hyper["message_pass_type"]).to(device)

            state_dims = [i.shape for i in model.parameters()]
            fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
            state_dim = len(fast_weights)
            hippo_model = models.HIPPO(state_dim=state_dim)

            model.train()
            hippo_model.train()

            optimizer = create_optimizer(
                opt=args.optimizer, 
                model=model,
                lr=hyper["lr"], 
                weight_decay=hyper["weight_decay"])

            n = math.ceil(len(graph_l) * 0.7)
            best_param = train(args=args, 
                            model=model,
                            hippo_model=hippo_model,
                            optimizer=optimizer, 
                            device=device, 
                            graph_l=graph_l[:n])

            model.load_state_dict(best_param['best_state_model'])
            hippo_model.load_state_dict(best_param["best_state_hippo"])

            model.eval()
            hippo_model.eval() 
            result = test(graph_l, model, hippo_model, args, n, device)

            if result["avg_mrr"] > best_mrr:
                best_graph_model = best_param["best_state_model"]
                best_state_hippo_model = best_param["best_state_hippo"]
            final_result["all_avg_mrr"].append(result["avg_mrr"])
            final_result["all_avg_macro_auc"].append(result["avg_macro_auc"])
            final_result["all_avg_micro_auc"].append(result["avg_micro_auc"])
            final_result["all_avg_f1"].append(result["avg_f1"])
            final_result["all_avg_acc"].append(result["avg_acc"])
            final_result["all_avg_ap"].append(result["avg_ap"])
            final_result["all_avg_rl10"].append(result["avg_rl10"])   
        print(final_result)
        res = pd.DataFrame({"metrics":final_result.keys(), "result":final_result.values()})
        res.to_csv(f"{path}/results/{args.dataset}/{dic_keys}.csv")
        torch.save(best_graph_model, f"{path}/best_models/{args.dataset}/best_graph_model_{args.dataset}_{dic_keys}.pkl")
        torch.save(best_state_hippo_model, f"{path}/best_models/{args.dataset}/best_state_hippo_model_{args.dataset}_{dic_keys}.pkl")

