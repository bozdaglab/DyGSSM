import torch
from copy import deepcopy
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent)
from config import cfg
from loss import Link_loss_meta, prediction
from utils_helper import report_rank_based_eval_meta
from Logger import getLogger
from tqdm import tqdm
logger = getLogger(cfg.log_path)



def train(args, 
            model,
            hippo_model,
            optimizer, 
            device, 
            graph_l):

    best_mrr = 0
    best_param = {'best_state_model': None, 
                  "best_state_hippo": None, 
                 }
    earl_stop_c = 0
    epoch_count = 0
    loss_avg = 0.0
    beta = 0.9                # EMA smoothing factor
    max_loss_scale = 0.1      # Maximum allowed scaling from loss
    max_gate = 1.0
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm(range(args.epochs)):

        all_mrr = 0.0
        fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
        window_mrr = 0.0
        count = 0
        for idx, graph in tqdm(enumerate(graph_l)):
            fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
            feature_train = deepcopy(graph_l[idx].node_feature).to(device)
            graph = graph.to(device)

            pred = model(graph, feature_train)
            loss = Link_loss_meta(pred, graph.edge_label)

            grad = torch.autograd.grad(loss, fast_weights)
            
            graph = graph.to(device)
            feature_train = feature_train.to(device)
            grad_vector = torch.tensor([torch.mean(g) for g in grad])
            weights = 1 / (loss.item()+ 1e-6) 
            hippo_state = hippo_model(grad_vector, weights)

            # # Inside loop
            # loss_value = loss.item()

            # if loss_avg is None:
            #     loss_avg = loss_value
            # else:
            #     loss_avg = beta * loss_avg + (1 - beta) * loss_value

            loss_scale = min(loss.item(), max_loss_scale)

            fast_weights = []
            for g, w, h in zip(grad, fast_weights, hippo_state):
                gate = torch.tanh(g * h)
                gate = torch.clamp(gate, -max_gate, max_gate)
                fast_weights.append(w - loss_scale * gate)

            # if not loss_initialized:
            #     loss_avg = loss
            #     loss_initialized = True
            # else:
            #     loss_avg = beta * loss_avg + (1 - beta) * loss.item()

            # # --- Clip the smoothed loss scale ---
            # loss_scale = min(loss_avg, max_loss_scale) / 2
            # fast_weights = list(
            #     map(lambda p: p[1] - loss_scale * torch.tanh(p[0] * p[2]), zip(grad, fast_weights, hippo_state))
            # )
            with torch.no_grad():
                for model_param, param in zip(model.parameters(), fast_weights):
                    model_param.copy_(param)

            if idx == len(graph_l) - 1:
                break
            graph_l[idx + 1] = graph_l[idx + 1]


            pred = model(graph_l[idx + 1].to(device), graph_l[idx + 1].node_feature.to(device))
            loss = Link_loss_meta(pred, graph_l[idx + 1].edge_label)

            edge_label = graph_l[idx + 1].edge_label
            edge_label_index = graph_l[idx + 1].edge_label_index
            mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_l[idx + 1], graph_l[idx+1].node_feature,
                                                                fast_weights, device)
            graph_l[idx + 1].edge_label = edge_label
            graph_l[idx + 1].edge_label_index = edge_label_index

            count += 1
            window_mrr += mrr
            acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_l[idx + 1].edge_label)
            logger.info('meta epoch:{}, mrr:{:.5f}, r@10:{:.5f}, loss_model: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                        format(epoch, mrr, rl10, loss, acc, ap, f1, macro_auc, micro_auc))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        all_mrr += window_mrr / len(graph_l)
        epoch_count += 1
        if all_mrr > best_mrr:
            best_mrr = all_mrr
            best_param = {'best_state_model': deepcopy(model.state_dict()), 
                          "best_state_hippo": deepcopy(hippo_model.state_dict()),
                          }
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break

    return best_param