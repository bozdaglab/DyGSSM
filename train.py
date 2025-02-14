import torch
from copy import deepcopy
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent)
from config import cfg
from loss import Link_loss_meta, prediction
from utils import report_rank_based_eval_meta
from Logger import getLogger
from tqdm import tqdm
logger = getLogger(cfg.log_path)



def train(args, 
            model,
            sequential_model, 
            cross_attention,
            hippo_model,
            optimizer, 
            optimizer_gru, 
            device, 
            graph_l, 
            n):

    best_mrr = 0
    best_param = {'best_state_model': None, 
                  "best_state_gru": None, 
                  "best_state_hippo": None, 
                  "best_state_transformer": None}
    earl_stop_c = 0
    epoch_count = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(args.epochs)):
        all_mrr = 0.0
        i = 0
        fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
        train_count = 0
        for i in range(n - args.window_num + 1):
            graph_train = graph_l[i : i + args.window_num]
            i = i + 1
            graph_train = graph_l[i : i + args.window_num]
            i = i + 1
            features = [graph_unit.node_feature for graph_unit in graph_train]

            fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
            window_mrr = 0.0
            losses = torch.tensor(0.0)
            loss_gru = torch.tensor(0.0)
            count = 0

            for idx, graph in tqdm(enumerate(graph_train)):
                torch.cuda.empty_cache()
                feature_train = deepcopy(features[idx]).to(device)
                graph = graph.to(device)

                pred_gru, seq_embeddings = sequential_model(graph, device)
                loss_sequence = Link_loss_meta(pred_gru, graph.edge_label)
                pred, _ = model(graph, feature_train, fast_weights, seq_embeddings, cross_attention)
                loss = Link_loss_meta(pred, graph.edge_label)

                grad = torch.autograd.grad(loss, fast_weights)
                graph = graph.to(device)
                feature_train = feature_train.to(device)
                grad_vector = torch.tensor([torch.mean(g) for g in grad])
                weights = 1 / (loss.item()+ 1e-6) 
                hippo_state = hippo_model(grad_vector, weights)
                fast_weights = list(
                    map(lambda p: p[1] - (loss.item()/2) * (p[0] * p[2]), zip(grad, fast_weights, hippo_state))
                )

                if idx == args.window_num - 1:
                    break
                graph_train[idx + 1] = graph_train[idx + 1]

                pred_gru, seq_embeddings = sequential_model(graph_train[idx + 1].to(device), device)
                loss_sequence = Link_loss_meta(pred_gru, graph_train[idx + 1].edge_label)

                pred, _ = model(graph_train[idx + 1].to(device), features[idx + 1].to(device), fast_weights, seq_embeddings, cross_attention)
                loss = Link_loss_meta(pred, graph_train[idx + 1].edge_label)

                edge_label = graph_train[idx + 1].edge_label
                edge_label_index = graph_train[idx + 1].edge_label_index
                mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_train[idx + 1], features[idx+1],
                                                                  fast_weights, sequential_model, cross_attention, device)
                graph_train[idx + 1].edge_label = edge_label
                graph_train[idx + 1].edge_label_index = edge_label_index

                losses = losses + loss
                loss_gru = loss_gru + loss_sequence
                count += 1
                window_mrr += mrr
                acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_train[idx + 1].edge_label)
                logger.info('meta epoch:{}, mrr:{:.5f}, r@10:{:.5f}, loss_model: {:.5f}, loss_gru:{:.5f} acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                            format(epoch, mrr, rl10, loss, loss_sequence, acc, ap, f1, macro_auc, micro_auc))
                torch.cuda.empty_cache()
            if losses:
                losses = losses / count
                loss_gru = loss_gru / count
                optimizer.zero_grad()
                losses.backward(retain_graph=True)
                optimizer_gru.zero_grad()
                loss_gru.backward()
            if count:
                all_mrr += window_mrr / count
                train_count += 1

            optimizer.step()
            optimizer_gru.step()
        all_mrr = all_mrr / train_count
        epoch_count += 1
        if all_mrr > best_mrr:
            best_mrr = all_mrr
            best_param = {'best_state_model': deepcopy(model.state_dict()), 
                          "best_state_gru": deepcopy(sequential_model.state_dict()), 
                          "best_state_hippo": deepcopy(hippo_model.state_dict()),
                          "best_state_cross_attention":deepcopy(cross_attention.state_dict())}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break

    return best_param