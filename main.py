import pandas as pd
import math
import torch
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import sys
sys.path.append(f"{Path(__file__).parent.parent}")
import models
from test import test
from train import train
from config import cfg
from Logger import getLogger
from utils import create_optimizer
from helper import load_data
from get_args import load_args
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = load_args()
    logger = getLogger(cfg.log_path)
    path_1 = str(Path(__file__).parent.parent.parent)
    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = 0
    final_result = defaultdict(list)
    for rep in range(0, args.repeat):
        path = str(Path(__file__).parent)
        dataset_name = args.dataset
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        graph_l = load_data(path, rep, args,path_1, device)

        n_dim = graph_l[0].node_feature.shape[1]
        n_node = graph_l[0].num_nodes()


        model = models.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout, args).to(device)
        if args.light_gru:
            sequential_model = models.GRUModel(
            input_dim=n_node,
            hidden_dim=args.num_hidden,
            output_dim=args.hidden_dim,
            num_layer=args.seq_num_layer,
            ).to(device)
        else:
            sequential_model = models.SeqRW(
                inp_size=n_node,
                emb_dim=args.num_hidden,
                hidden=args.hidden_dim,
                num_layers=args.seq_num_layer,
                dropout=args.dropout_1,
            ).to(device)
        cross_attention = models.CrossAttention(
            embed_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout_1,
            num_cross_att=1,
        ).to(device)

        state_dims = [i.shape for i in model.parameters()]
        fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
        state_dim = len(fast_weights)
        hippo_model = models.HIPPO(state_dim=state_dim)

        model.train()
        sequential_model.train()
        # cross_attention.train()
        hippo_model.train()

        optimizer, optimizer_gru = create_optimizer(
            opt=args.optimizer, 
            cross_attention=cross_attention,
            model=model,
            sequential_model=sequential_model, 
            lr=args.lr, 
            weight_decay=args.weight_decay)

        n = math.ceil(len(graph_l) * 0.7)
        best_param = train(args=args, 
                           model=model,
                           sequential_model=sequential_model, 
                           cross_attention=cross_attention,
                           hippo_model=hippo_model,
                           optimizer=optimizer, 
                           optimizer_gru=optimizer_gru, 
                           device=device, 
                           graph_l=graph_l, 
                           n=n)

        model.load_state_dict(best_param['best_state_model'])
        sequential_model.load_state_dict(best_param["best_state_gru"])
        hippo_model.load_state_dict(best_param["best_state_hippo"])
        cross_attention.load_state_dict(best_param["best_state_cross_attention"])

        model.eval()
        sequential_model.eval()
        cross_attention.eval()
        hippo_model.eval() 
        result = test(graph_l, model, sequential_model, cross_attention, hippo_model, args, n, device)

        if result["avg_mrr"] > best_mrr:
            best_graph_model = best_param["best_state_model"]
            best_graph_gru = best_param["best_state_gru"]
            best_state_hippo_model = best_param["best_state_hippo"]
            best_state_cross_attention = best_param["best_state_cross_attention"]
        final_result["all_avg_mrr"].append(result["avg_mrr"])
        final_result["all_avg_macro_auc"].append(result["avg_macro_auc"])
        final_result["all_avg_micro_auc"].append(result["avg_micro_auc"])
        final_result["all_avg_f1"].append(result["avg_f1"])
        final_result["all_avg_acc"].append(result["avg_acc"])
        final_result["all_avg_ap"].append(result["avg_ap"])
        final_result["all_avg_rl10"].append(result["avg_rl10"])   
    print(final_result)
    res = pd.DataFrame({"metrics":final_result.keys(), "result":final_result.values()})
    res.to_csv(f"{path}/{args.dataset}_{args.lr}_{args.window_num}_{args.hidden_dim}_beta_update_overlapped_wind_no_gru_update_no_transformer_replaced_maml_lr_loss.csv")
    torch.save(best_graph_model, f"{path}/best_graph_model_{args.dataset}_no_transformer_replaced_maml_lr_loss.pkl")
    torch.save(best_graph_gru, f"{path}/best_graph_gru_{args.dataset}_no_transformer_replaced_maml_lr_loss.pkl")
    torch.save(best_state_cross_attention, f"{path}/best_state_cross_attention_{args.dataset}_no_transformer_replaced_maml_lr_loss.pkl")
    torch.save(best_state_hippo_model, f"{path}/best_state_hippo_model_{args.dataset}_no_transformer_replaced_maml_lr_loss.pkl")

