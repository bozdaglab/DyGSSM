#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import numpy as np
import torch
from copy import deepcopy
from loss import prediction, Link_loss_meta
from utils_helper import report_rank_based_eval_meta


def test(graph_l, model, hippo_model, args, n, device):

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
        graph_train = deepcopy(g_test.node_feature).to(device)
        graph_train = graph_train.to(device)
        g_test = g_test.to(device)
        pred = model(g_test, graph_train)
        edge_label = graph_test[idx].edge_label
        edge_label_index = graph_test[idx].edge_label_index
        mrr, rl1, rl3, rl10 = report_rank_based_eval_meta(model, graph_test[idx], graph_test[idx].node_feature,
                                                          fast_weights, device)
        graph_test[idx].edge_label = edge_label
        graph_test[idx].edge_label_index = edge_label_index
        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_test[idx].edge_label)
        result["avg_mrr"] += mrr
        result["avg_macro_auc"] += macro_auc
        result["avg_micro_auc"] += micro_auc
        result["avg_f1"] += f1
        result["avg_acc"] += acc
        result["avg_ap"] += ap
        result["avg_rl10"] += rl10
        print('meta test, mrr: {:.5f}, rl1: {:.5f}, rl3: {:.5f}, rl10: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                    format(mrr, rl1, rl3, rl10, acc, ap, f1, macro_auc, micro_auc))

    result["avg_mrr"] /= len(graph_test)
    result["avg_macro_auc"] /= len(graph_test)
    result["avg_micro_auc"] /= len(graph_test)
    result["avg_f1"] /= len(graph_test)
    result["avg_acc"] /= len(graph_test)
    result["avg_ap"] /= len(graph_test)
    result["avg_rl10"] /= len(graph_test)
    print({
        "avg_mrr": result["avg_mrr"],
        "avg_macro_auc": result["avg_macro_auc"],
        "avg_micro_auc": result["avg_micro_auc"],
        "avg_f1": result["avg_f1"],
        "avg_acc": result["avg_acc"],
        "avg_ap": result["avg_ap"],
        "avg_rl10": result["avg_rl10"]})
    return result
