#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

def prediction(pred_score, true_l):
    pred = pred_score.clone()
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()
    true = true_l
    true = true.cpu().numpy()
    acc = accuracy_score(true, pred)
    ap = average_precision_score(true, pred_score)
    f1 = f1_score(true, pred, average='macro')
    macro_auc = roc_auc_score(true, pred_score, average='macro')
    micro_auc = roc_auc_score(true, pred_score, average='micro')
    return acc, ap, f1, macro_auc, micro_auc


def Link_loss_meta(pred, y):
    L = nn.BCELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)
    return loss


def Link_loss_meta_mse(pred, y):
    L = nn.MSELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)
    return loss

def node2vec_loss(seq_embeddings, pos_rw, neg_rw):
    r"""Computes the loss given positive and negative random walks."""
    # Positive loss.
    eps = 1e-15
    start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

    h_start = seq_embeddings[start].view(pos_rw.size(0), 1, seq_embeddings.shape[1])
    h_rest = seq_embeddings[rest.view(-1)].view(pos_rw.size(0), -1,seq_embeddings.shape[1])

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    pos_loss = -torch.log(torch.sigmoid(out) + eps).mean()

    # Negative loss.
    start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

    h_start = seq_embeddings[start].view(neg_rw.size(0), 1,seq_embeddings.shape[1])
    h_rest = seq_embeddings[rest.view(-1)].view(neg_rw.size(0), -1,seq_embeddings.shape[1])

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    neg_loss = -torch.log(1 - torch.sigmoid(out) + eps).mean()
    return pos_loss + neg_loss