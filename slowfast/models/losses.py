#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def distributed_sinkhorn(Q, nmb_iters, cfg):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (cfg.world_size * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def swav_loss_wrapper(queue = None, reduction=None):
    # TODO crosscheck
    return swav_loss

def swav_loss(output, cfg, bs = None, queue=None, reduction=None):
    loss = 0
    softmax = nn.Softmax(dim=1).cuda()
    print("In swav_loss")
    print(output.shape)

    for i, crop_id in enumerate(cfg.SWAV_crops_for_assign):
        with torch.no_grad():
            out = output[bs * crop_id: bs * (crop_id + 1)]
            # time to use the queue
            # TODO implement queue
            # if queue is not None:
            #     if use_the_queue or not torch.all(queue[i, -1, :] == 0):
            #         use_the_queue = True
            #         out = torch.cat((torch.mm(
            #             queue[i],
            #             model.module.prototypes.weight.t()
            #         ), out))
            #     # fill the queue
            #     queue[i, bs:] = queue[i, :-bs].clone()
            #     queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
            # get assignments
            q = out / cfg.SWAV_epsilon
            # TODO improve_numerical_stability
            # if cfg.improve_numerical_stability:
            #     M = torch.max(q)
            #     dist.all_reduce(M, op=dist.ReduceOp.MAX)
            #     q -= M
            q = torch.exp(q).t()
            q = distributed_sinkhorn(q, cfg.SWAV_sinkhorn_iterations, cfg)[-bs:]

        # cluster assignment prediction
        subloss = 0
        if cfg.SWAV_nmb_crops > 0:
            for v in np.delete(np.arange(np.sum(cfg.SWAV_nmb_crops)+cfg.SWAV_nmb_frame_views), crop_id):
                p = softmax(output[bs * v: bs * (v + 1)] / cfg.SWAV_temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(cfg.SWAV_nmb_crops) - 1)
    loss /= len(cfg.SWAV_crops_for_assign)
    return loss

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "swav_loss": swav_loss
}
