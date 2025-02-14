#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import torch
from copy import deepcopy
from DyGSSM.loss import prediction, Link_loss_meta
from DyGSSM.utils import report_rank_based_eval_meta


def test(graph_l, model, sequential_model, cross_attention, hippo_model, args, n, device):

    result = {
        "avg_mrr": 0.0,
        "avg_macro_auc": 0.0,
        "avg_micro_auc": 0.0,
        "avg_f1": 0.0,
        "avg_acc": 0.0,
        "avg_ap": 0.0,
        "avg_rl10": 0.0
    }

    graph_test = graph_l[n:]
    fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
    for idx, g_test in enumerate(graph_test):

        if idx == len(graph_test) - 1:
            break

        graph_train = deepcopy(g_test.node_feature).to(device)
        graph_train = graph_train.to(device)
        g_test = g_test.to(device)
        _, seq_embeddings = sequential_model(g_test, device)
        pred, _ = model(g_test, graph_train, fast_weights, seq_embeddings, cross_attention)
        loss = Link_loss_meta(pred, g_test.edge_label)

        graph_train = graph_train
        grad = torch.autograd.grad(loss, fast_weights)
        g_test = g_test
        grad_vector = torch.tensor([torch.mean(g) for g in grad])
        weights = 1 / (loss.item()+ 1e-6)
        hippo_state = hippo_model(grad_vector, weights)
        fast_weights = list(
            map(lambda p: p[1] - args.maml_lr * (p[0] * p[2]), zip(grad, fast_weights, hippo_state))
        )

        graph_test[idx + 1] = graph_test[idx + 1]
        graph_test[idx + 1].node_feature = graph_test[idx + 1].node_feature
        _, seq_embeddings = sequential_model(graph_test[idx + 1].to(device), device)
        pred, _ = model(graph_test[idx + 1].to(device), graph_test[idx + 1].node_feature.to(device), fast_weights, seq_embeddings, cross_attention)

        loss = Link_loss_meta(pred, graph_test[idx + 1].edge_label)

        edge_label = graph_test[idx + 1].edge_label
        edge_label_index = graph_test[idx + 1].edge_label_index
        mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_test[idx + 1], graph_test[idx + 1].node_feature,
                                                          fast_weights, sequential_model, cross_attention, device)
        graph_test[idx + 1].edge_label = edge_label
        graph_test[idx + 1].edge_label_index = edge_label_index

        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_test[idx + 1].edge_label)

        result["avg_mrr"] += mrr
        result["avg_macro_auc"] += macro_auc
        result["avg_micro_auc"] += micro_auc
        result["avg_f1"] += f1
        result["avg_acc"] += acc
        result["avg_ap"] += ap
        result["avg_rl10"] += rl10
        print('meta test, mrr: {:.5f}, rl1: {:.5f}, rl3: {:.5f}, rl10: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                    format(mrr, rl1, rl3, rl10, acc, ap, f1, macro_auc, micro_auc))

    result["avg_mrr"] /= len(graph_test) - 1
    result["avg_macro_auc"] /= len(graph_test) - 1
    result["avg_micro_auc"] /= len(graph_test) - 1
    result["avg_f1"] /= len(graph_test) - 1
    result["avg_acc"] /= len(graph_test) - 1
    result["avg_ap"] /= len(graph_test) - 1
    result["avg_rl10"] /= len(graph_test) - 1
    print({
        "avg_mrr": result["avg_mrr"],
        "avg_macro_auc": result["avg_macro_auc"],
        "avg_micro_auc": result["avg_micro_auc"],
        "avg_f1": result["avg_f1"],
        "avg_acc": result["avg_acc"],
        "avg_ap": result["avg_ap"],
        "avg_rl10": result["avg_rl10"]})
    return result
