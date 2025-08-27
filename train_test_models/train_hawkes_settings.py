import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from hawkesGNN.utils import seed_everything, generate_random_seeds, save_result
from hawkesGNN.models.base import BaseLPModel
import copy
import time
from hawkesGNN.utils import make_negative_adj,  calculate_row_mrr, calculate_sample_mrr, make_full_adj
from hawkesGNN.train import LinkPrediction
import random 
from deepsnap.graph import Graph
import dgl
from helper import make_usci_data
import pickle
import os
from pathlib import Path

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

import random
from collections import Counter
import dgl
from torch_geometric.nn import Node2Vec

def check_rw_needed(prev, row, col):
    try:
        set1 = set(map(tuple, torch.stack([torch.from_numpy(row), torch.from_numpy(col)]).t().tolist()))
    except TypeError:
        set1 = set(map(tuple, torch.stack([row, col]).t().tolist()))
    set2 = set(map(tuple, prev.tolist()))
    effected_edges = set1 - set2
    return torch.tensor(list(effected_edges)).t(), True if len(effected_edges)> 0 else False

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

class DyGSSMLinkPrediction(LinkPrediction):
    
    @torch.no_grad()
    def test_snap(self, model, fast_weights, snaps, device, return_full_mrr=False):
        loss_list = []
        result = [0] * 7

        data, target = snaps[0], snaps[1]
        h = model(data, data.node_feature)

        pos_edges, neg_edges, idx = self.prepare_test_edges(target, device)
        h, pos_out, neg_out, loss = model.test_step(h, pos_edges, neg_edges, idx)
        loss_list.append(loss.item())
        
        res = calculate_sample_mrr(pos_out, neg_out, self.n_neg_test)
        for i, r in enumerate(res):
            result[i+1] = r.item()

        # if return_full_mrr:   # accelate training process
        #     full_adj = make_full_adj(target)
        #     full_out = model.predict(h, full_adj.indices().to(device))
        #     mrr = calculate_row_mrr(target, full_adj, full_out, device)
        #     result[0] = mrr

        # loss, [mrr@row, mrr@1000, hit@1, hit@3, hit@10]
        return np.mean(loss_list), result
    
 
    
    def train_epoch(self, args, model:BaseLPModel, hippo_model, S_dw, ds_train,datasets, device, optimizer):
        model.train()
        max_loss_scale = 0.1      # Maximum allowed scaling from loss
        max_gate = 1.0
        fast_weights = list(model.parameters())
        mrr_list = []
        for idx, data in enumerate(datasets[:-2]):
            fast_weights = list(model.parameters())
            target = ds_train[idx+1]

            h = model(data.to(device), data.node_feature.to(device))
            pos_edges, neg_edges = self.negative_sampling(target, device)
            loss = model.train_step(h, pos_edges, neg_edges)

            grad = torch.autograd.grad(loss, fast_weights)
            grad_vector = torch.tensor([torch.mean(g) for g in grad])
            weights = 1 / (loss.item()+ 1e-6) 
            hippo_state = hippo_model(grad_vector, weights)
            loss_scale = min(loss.item(), max_loss_scale)

            fast_weights = []
            for g, w, h in zip(grad, fast_weights, hippo_state):
                gate = torch.tanh(g * h)
                gate = torch.clamp(gate, -max_gate, max_gate)
                fast_weights.append(w - loss_scale * gate)
                
            with torch.no_grad():
                for model_param, param in zip(model.parameters(), fast_weights):
                    model_param.copy_(param)
            # S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0] * p[0], zip(grad, S_dw)))
            # fast_weights = list(
            #         map(lambda p: p[1] - args.DyGSSM_maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

            # official implementaion is wrong! 
            # we are not to predict the input edge_index
            # but the future one
            snaps = (datasets[idx+1], ds_train[idx+2]) 
            data, target = snaps
            h = model(data.to(device), data.node_feature.to(device))
            pos_edges, neg_edges = self.negative_sampling(target, device)
            val_loss = model.train_step(h, pos_edges, neg_edges)
            _, mets = self.test_snap(model, fast_weights, snaps, device, False)
            

            # losses = val_loss  #!!!!! so wired, sgd using val loss
                # mrr_window_list.append(mets[1])


            optimizer.zero_grad()
            val_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()
            mrr_list.append(mets[1])

        return np.mean(mrr_list)
    

    def test(self, args, model, hippo_model, S_dw, dataset, dataset_all, split, device):
        ds_test = dataset_all[split[1]-2:]
        ds_dataset_test = dataset[split[1]-2:]
        fast_weights = list(model.parameters())
        loss_list = []
        result_list = []
        count_list = []
        max_loss_scale = 0.1      # Maximum allowed scaling from loss
        max_gate = 1.0
        for idx, data in enumerate(ds_test[:-2]):
            fast_weights = list(model.parameters())
            target = ds_dataset_test[idx+1]
            model.train()
            h = model(data, data.node_feature)
            pos_edges, neg_edges, reidx = self.prepare_test_edges(target, args.device)
            h, pos_out, neg_out, loss = model.test_step(h, pos_edges, neg_edges, reidx)

            # grad = torch.autograd.grad(loss, fast_weights)
            # fast_weights = list(map(lambda p: p[1] - args.DyGSSM_maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))
            grad = torch.autograd.grad(loss, fast_weights)
            grad_vector = torch.tensor([torch.mean(g) for g in grad])
            weights = 1 / (loss.item()+ 1e-6) 
            hippo_state = hippo_model(grad_vector, weights)
            loss_scale = min(loss.item(), max_loss_scale)

            fast_weights = []
            for g, w, h in zip(grad, fast_weights, hippo_state):
                gate = torch.tanh(g * h)
                gate = torch.clamp(gate, -max_gate, max_gate)
                fast_weights.append(w - loss_scale * gate)
                
            with torch.no_grad():
                for model_param, param in zip(model.parameters(), fast_weights):
                    model_param.copy_(param)
            # official implementaion is wrong! links to predict must be next snap!
            snaps = (ds_test[idx+1], ds_dataset_test[idx+2]) 
            model.eval()
            test_loss, mets = self.test_snap(model, fast_weights, snaps, device, self.n_neg_test)
            loss_list.append(test_loss.item())
            result_list.append(mets)
            count_list.append(len(pos_out))
        
        result = np.array(result_list)
        count = np.array(count_list)
        return np.mean(loss_list), (result * count.reshape(-1, 1)).sum(0) / count.sum()
    
        

    def train(self, args, model, hippo, dataset, dataset_all, split, device):
        S_dw = [0] * len(list(model.parameters()))
        
        # https://github.com/pytorch/pytorch/issues/113758
        optimizer = torch.optim.AdamW(params=model.parameters(),weight_decay=args.weight_decay,lr=args.lr, foreach=False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        best_mrr = 0
        wandering = 0
        state_dict = None
        time_usage_list = []
                
        for epoch in tqdm(range(1, 1 + args.epochs)):
            t0 = time.perf_counter()
            valid_mrr = self.train_epoch(args, model, hippo, S_dw, dataset[:split[1]],dataset_all[:split[1]], device, optimizer)
            t1 = time.perf_counter()
            print(valid_mrr)
            # lr_scheduler.step()

            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                wandering = 0
                state_dict = copy.deepcopy(model.state_dict())
            else:
                wandering += 1
            
            if wandering > args.patiance:
                break
            time_usage_list.append(t1-t0)
        
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model, S_dw, time_usage_list



    def main(self, args, factory, ds_split, device, path_1, model_type):
        if not os.path.exists(f"{path_1}/results_{model_type}/{args.dataset}"):
            os.makedirs(f"{path_1}/results_{model_type}/{args.dataset}")
            os.makedirs(f"{path_1}/processed_data_{model_type}/{args.dataset}")
        ds, split = ds_split[0], ds_split[1]
        mrr_lists = []
        random_seeds = generate_random_seeds(seed=args.seed, nums=args.runs)
        gpu_usage_list = []
        time_usage_list = []
        for run in range(args.runs):
            if os.path.exists(f"{path_1}/processed_data_{model_type}/{args.dataset}/{run}.pkl"):
                with open(f"{path_1}/processed_data_{model_type}/{args.dataset}/{run}.pkl", "rb") as file:
                    dataset_all = pickle.load(file)
            else:
                dataset_all = make_usci_data(graphs=ds, device=device, hawkes=True)
                with open(f"{path_1}/processed_data_{model_type}/{args.dataset}/{run}.pkl", "wb") as file:
                    pickle.dump(dataset_all, file)

            seed_everything(random_seeds[run])
            model, hippo = self.build_model(args, factory, device)
            model, S_dw, epoch_time = self.train(args, model, hippo, ds, dataset_all, split, device)
            test_loss, mrr = self.test(args, model, hippo, S_dw, ds, dataset_all, split, device)
            print(f'Test Metric: {mrr[1]:.4f}, {mrr[2]:.4f}, {mrr[3]:.4f}, {mrr[4]:.4f}')
            mrr_lists.append(mrr)
            time_usage_list.append(epoch_time)
            gpu_mem_alloc = torch.cuda.max_memory_allocated(args.device) / 1000000
            gpu_usage_list.append(gpu_mem_alloc)

        pd.DataFrame(
            np.stack(mrr_lists), columns=["nothing", "MRR", "hits@1","hits@3", "hits@10", "auc", "ap"]
            ).to_csv( f"{path_1}/results_{model_type}/{args.dataset}/results.csv")
        