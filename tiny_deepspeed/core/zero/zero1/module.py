# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist

from ...module import (
    ops,
    linear, 
    normalization, 
    embedding,
)
from .utils import Parameter

def sync_grad(grad, async_op=True, rank_id=None):    # communication complexity: g
    if async_op:
        return dist.reduce(grad, dst=rank_id, async_op=True)
    else:
        dist.reduce(grad, dst=rank_id, async_op=False)
        return None


class Linear(linear.Linear):
    def _init_parameters(self):
        self.weight = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
        if self.use_bias:
            self.bias = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def backward_callback(self, ctx, grad_output, runtime_tuner):
        input, weight, bias = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_weight = ops.linear_weight_grad(grad_output, input, weight, runtime_tuner)
            if self.weight.bwd_sync:    # core step of zero1
                handle_weight = sync_grad(grad_weight, rank_id=self.weight.rank_id)
                self.weight.bwd_sync = False
            else:
                handle_weight = None
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = ops.linear_bias_grad(grad_output, input, weight, runtime_tuner)
            if self.bias.bwd_sync:  # core step of zero1
                handle_bias = sync_grad(grad_bias, rank_id=self.bias.rank_id)
                self.bias.bwd_sync = False
            else:
                handle_bias = None
        else:
            grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = ops.linear_input_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_input = None

        # Communication-computation overlap, wait for the communication to finish (core step of zero1)
        if ctx.needs_input_grad[1] and handle_weight is not None:
            handle_weight.wait()
        if bias is not None and ctx.needs_input_grad[2] and handle_bias is not None:
            handle_bias.wait()

        # Check if the grad shape is correct
        if grad_input is not None and grad_input.shape != input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
        if grad_bias is not None and grad_bias.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")
        
        return grad_input, grad_weight, grad_bias


class LayerNorm(normalization.LayerNorm):
    def _init_parameters(self):
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
            if self.use_bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def backward_callback(self, ctx, grad_output, eps, runtime_tuner):
        input, weight, bias, mean, rstd = ctx.saved_tensors
        args = {
            'BLOCK_SIZE': ctx.args['BLOCK_SIZE'],
            'num_warps': ctx.args['num_warps'],
            'eps': eps,
        }
        dx, dw_, db_, args = ops.layernorm_dx(grad_output, input, weight, bias, mean, rstd, args, runtime_tuner)
        dw, db = ops.layernorm_dwdb(weight, bias, dw_, db_, args, runtime_tuner)
        if self.weight.bwd_sync:    # core step of zero1
            sync_grad(dw, async_op=False, rank_id=self.weight.rank_id)
            self.weight.bwd_sync = False
        if self.bias.bwd_sync:  # core step of zero1
            sync_grad(db, async_op=False, rank_id=self.bias.rank_id)
            self.bias.bwd_sync = False
        
        # Check if the grad shape is correct
        if dx is not None and dx.shape != input.shape:
            raise RuntimeError(f"grad_input shape {dx.shape} is not equal to input shape {input.shape}")
        if dw is not None and dw.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {dw.shape} is not equal to weight shape {weight.shape}")
        if db is not None and db.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {db.shape} is not equal to bias shape {bias.shape}")

        return dx, dw, db


class Embedding(embedding.Embedding):
    def _init_parameters(self):
        if self._weight is None:
            self.weight = Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **self.factory_kwargs),
                                    requires_grad=not self._freeze)
            self.reset_parameters()
        else:
            assert list(self._weight.shape) == [self.num_embeddings, self.embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(self._weight, requires_grad=not self._freeze)

    def backward_callback(self, ctx, grad_output, padding_idx, max_norm, norm_type, runtime_tuner):
        input, weight = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_weight = ops.embedding_weight_grad(grad_output, input, weight, runtime_tuner)
            if self.weight.bwd_sync:    # core step of zero1
                sync_grad(grad_weight, async_op=False, rank_id=self.weight.rank_id)
                self.weight.bwd_sync = False
        else:
            grad_weight = None

        # Check if the grad shape is correct
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")

        return grad_weight

