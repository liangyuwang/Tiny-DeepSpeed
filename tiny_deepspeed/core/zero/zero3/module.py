# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
import torch.distributed as dist
import math
import os

from ...module import (
    ops,
    linear, 
    normalization, 
    embedding,
)
from .utils import Parameter

def sync_param(param, async_op=False, rank_id=None):    # communication complexity: g
    current_rank = dist.get_rank()
    if current_rank != rank_id:
        # Create zero tensor with correct size and device for non-owner ranks
        # Use zeros instead of empty to avoid uninitialized memory with inf values
        param.data = torch.zeros(param.cached_size, dtype=param.cached_dtype, device=f'cuda:{current_rank}')
    
    # Ensure param.data is on the correct device
    if param.data.device != torch.device(f'cuda:{current_rank}'):
        param.data = param.data.to(f'cuda:{current_rank}')
    
    if async_op:
        return dist.broadcast(param.data, src=rank_id, async_op=True)
    else:
        dist.broadcast(param.data, src=rank_id, async_op=False)
        return None

# def desync_param(param, rank_id=None):
#     if rank_id:
#         if dist.get_rank() != rank_id:
#             param.data = torch.randn(1, device=param.device, dtype=param.dtype)
#             return param
#         else:
#             return param
#     return param

def desync_init(init_fn, rank_id, *args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'
    if dist.get_rank() != rank_id:
        # create empty tensor for other rank
        return torch.empty(0, device=device).clone()
    else:
        # create tensor for owner rank
        tensor = init_fn(*args, **kwargs).to(device)
        return tensor.clone()

def desync_param_data(param, rank_id):
    if dist.get_rank() != rank_id:
        # Instead of setting to None, create a small dummy tensor
        param.data = torch.empty(0, device=param.device, dtype=param.dtype)
    return param

def sync_grad(grad, async_op=True, rank_id=None):    # communication complexity: g
    if async_op:
        work = dist.reduce(grad, dst=rank_id, async_op=True)
    else:
        dist.reduce(grad, dst=rank_id, async_op=False)
        work = None
    torch.cuda.synchronize()
    return work

def desync_grad(grad, rank_id):
    if grad is not None and rank_id is not None:
        if dist.get_rank() != rank_id:
            # Return None for non-owner ranks
            return None
        return grad
    else:
        return None


class Linear(linear.Linear):
    def _init_parameters(self):
        with torch.device('meta'):  # Fake init
            self.weight = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
            if self.use_bias:
                self.bias = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
            else:
                self.register_parameter('bias', None)
    
    def reinit_parameters(self):
        # Save rank_ids before recreating parameters
        weight_rank_id = self.weight.rank_id
        bias_rank_id = self.bias.rank_id if self.bias is not None else None
        
        # Create new weight parameter
        weight_tensor = desync_init(
            torch.empty, weight_rank_id, (self.out_features, self.in_features), 
            dtype=self.factory_kwargs.get('dtype', torch.float32))
        self.weight = Parameter(weight_tensor)
        self.weight.rank_id = weight_rank_id  # Restore rank_id
        
        if self.use_bias:
            # Create new bias parameter
            bias_tensor = desync_init(
                torch.empty, bias_rank_id, self.out_features, 
                dtype=self.factory_kwargs.get('dtype', torch.float32))
            self.bias = Parameter(bias_tensor)
            self.bias.rank_id = bias_rank_id  # Restore rank_id
        else:
            self.register_parameter('bias', None)
        self.reset_parameters_fn()
    
    def reset_parameters_fn(self) -> None:
        if dist.get_rank() == self.weight.rank_id:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None and dist.get_rank() == self.bias.rank_id:
            # Calculate fan_in from weight dimensions directly to avoid accessing empty weight
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        ctx.save_for_backward(input, weight, bias)
        
        # Sync parameters for forward computation (core step of zero3)
        sync_param(weight, rank_id=weight.rank_id)
        if bias is not None:
            sync_param(bias, rank_id=bias.rank_id)
            
        output = ops.linear_forward(input, weight, bias, runtime_tuner)
        
        # Immediately desync after forward to free memory (core step of zero3)
        desync_param_data(weight, rank_id=weight.rank_id)
        if bias is not None:
            desync_param_data(bias, rank_id=bias.rank_id)
            
        return ctx, output

    def backward_callback(self, ctx, grad_output, runtime_tuner):
        input, weight, bias = ctx.saved_tensors

        # Sync parameters for backward computation (core step of zero3)
        sync_param(weight, rank_id=self.weight.rank_id)
        if bias is not None:
            sync_param(bias, rank_id=self.bias.rank_id)

        # Compute weights gradients
        if ctx.needs_input_grad[1]:
            grad_weight = ops.linear_weight_grad(grad_output, input, weight, runtime_tuner)
            if self.weight.bwd_sync:
                handle_weight = sync_grad(grad_weight, rank_id=self.weight.rank_id)
            else:
                handle_weight = None
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = ops.linear_bias_grad(grad_output, input, weight, runtime_tuner)
            if self.bias.bwd_sync:
                handle_bias = sync_grad(grad_bias, rank_id=self.bias.rank_id)
            else:
                handle_bias = None
        else:
            grad_bias = None
        
        # Compute input gradients
        grad_input = ops.linear_input_grad(grad_output, input, weight, runtime_tuner) if ctx.needs_input_grad[0] else None

        # Communication-computation overlap, wait for the communication to finish
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
        
        # Desync the grad and params for non-owner ranks
        if ctx.needs_input_grad[1] and hasattr(self.weight, 'bwd_sync') and self.weight.bwd_sync:
            grad_weight = desync_grad(grad_weight, rank_id=self.weight.rank_id)
        if bias is not None and ctx.needs_input_grad[2] and hasattr(self.bias, 'bwd_sync') and self.bias.bwd_sync:
            grad_bias = desync_grad(grad_bias, rank_id=self.bias.rank_id)
            
        # Desync parameters after backward
        desync_param_data(weight, rank_id=self.weight.rank_id)
        if bias is not None:
            desync_param_data(bias, rank_id=self.bias.rank_id)

        return grad_input, grad_weight, grad_bias


class LayerNorm(normalization.LayerNorm):
    def _init_parameters(self):
        with torch.device('meta'):  # Fake init
            if self.elementwise_affine:
                self.weight = Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
                if self.use_bias:
                    self.bias = Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
                else:
                    self.register_parameter('bias', None)
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
    
    def reinit_parameters(self):
        if self.elementwise_affine:
            # Save rank_ids before recreating parameters
            weight_rank_id = self.weight.rank_id
            bias_rank_id = self.bias.rank_id if self.bias is not None else None
            
            # Create new weight parameter
            weight_tensor = desync_init(
                torch.empty, weight_rank_id, self.normalized_shape, 
                dtype=self.factory_kwargs.get('dtype', torch.float32))
            self.weight = Parameter(weight_tensor)
            self.weight.rank_id = weight_rank_id  # Restore rank_id
            
            if self.use_bias:
                # Create new bias parameter
                bias_tensor = desync_init(
                    torch.empty, bias_rank_id, self.normalized_shape, 
                    dtype=self.factory_kwargs.get('dtype', torch.float32))
                self.bias = Parameter(bias_tensor)
                self.bias.rank_id = bias_rank_id  # Restore rank_id
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters_fn()
    
    def reset_parameters_fn(self) -> None:
        if self.elementwise_affine:
            if dist.get_rank() == self.weight.rank_id:
                nn.init.ones_(self.weight)
            if self.bias is not None and dist.get_rank() == self.bias.rank_id:
                nn.init.zeros_(self.bias)

    def forward_callback(self, ctx, input, weight, bias, eps, runtime_tuner):
        # Sync parameters for forward computation (core step of zero3)
        sync_param(weight, rank_id=weight.rank_id)
        if bias is not None:
            sync_param(bias, rank_id=bias.rank_id)
            
        output, mean, rstd, args = ops.layernorm_fwd(input, weight, bias, eps, runtime_tuner)
        
        # Immediately desync after forward (core step of zero3)
        desync_param_data(weight, rank_id=weight.rank_id)
        if bias is not None:
            desync_param_data(bias, rank_id=bias.rank_id)
            
        ctx.save_for_backward(input, weight, bias, mean, rstd)
        ctx.args = args
        return ctx, output

    def backward_callback(self, ctx, grad_output, eps, runtime_tuner):
        input, weight, bias, mean, rstd = ctx.saved_tensors
        
        # Sync parameters for backward computation (core step of zero3)
        sync_param(weight, rank_id=self.weight.rank_id)
        if bias is not None:
            sync_param(bias, rank_id=self.bias.rank_id)
            
        args = {
            'BLOCK_SIZE': ctx.args['BLOCK_SIZE'],
            'num_warps': ctx.args['num_warps'],
            'eps': eps,
        }
        dx, dw_, db_, args = ops.layernorm_dx(grad_output, input, weight, bias, mean, rstd, args, runtime_tuner)
        dw, db = ops.layernorm_dwdb(weight, bias, dw_, db_, args, runtime_tuner)
        if self.weight.bwd_sync:
            sync_grad(dw, async_op=False, rank_id=self.weight.rank_id)
        if self.bias.bwd_sync:
            sync_grad(db, async_op=False, rank_id=self.bias.rank_id)
        
        # Check if the grad shape is correct
        if dx is not None and dx.shape != input.shape:
            raise RuntimeError(f"grad_input shape {dx.shape} is not equal to input shape {input.shape}")
        if dw is not None and dw.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {dw.shape} is not equal to weight shape {weight.shape}")
        if db is not None and db.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {db.shape} is not equal to bias shape {bias.shape}")

        # Desync the grad for non-owner ranks
        if hasattr(self.weight, 'bwd_sync') and self.weight.bwd_sync:
            dw = desync_grad(dw, rank_id=self.weight.rank_id)
        if hasattr(self.bias, 'bwd_sync') and self.bias.bwd_sync:
            db = desync_grad(db, rank_id=self.bias.rank_id)
            
        # Desync parameters after backward
        desync_param_data(weight, rank_id=self.weight.rank_id)
        if bias is not None:
            desync_param_data(bias, rank_id=self.bias.rank_id)

        return dx, dw, db


class Embedding(embedding.Embedding):
    def _init_parameters(self):
        with torch.device('meta'):  # Fake init
            if self._weight is None:
                self.weight = Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **self.factory_kwargs),
                                        requires_grad=not self._freeze)
            else:
                raise NotImplementedError("Pretrained Embedding weight is not supported yet.")

    def reinit_parameters(self):
        if self._weight is None:
            # Save rank_id before recreating parameter
            weight_rank_id = self.weight.rank_id
            
            # Create new weight parameter
            weight_tensor = desync_init(torch.empty, weight_rank_id, (self.num_embeddings, self.embedding_dim), 
                                       dtype=self.factory_kwargs.get('dtype', torch.float32))
            self.weight = Parameter(weight_tensor, requires_grad=not self._freeze)
            self.weight.rank_id = weight_rank_id  # Restore rank_id
            
            self.reset_parameters()
        else:
            raise NotImplementedError("Pretrained Embedding weight is not supported yet.")

    def forward_callback(self, ctx, input, weight, padding_idx, max_norm, norm_type, runtime_tuner):
        ctx.save_for_backward(input, weight)
        
        # Sync parameters for forward computation (core step of zero3)
        sync_param(weight, rank_id=weight.rank_id)
        output = ops.embedding_forward(input, weight, padding_idx, max_norm, norm_type, runtime_tuner)
        
        # Immediately desync after forward (core step of zero3)
        desync_param_data(weight, rank_id=weight.rank_id)
        return ctx, output

    def backward_callback(self, ctx, grad_output, padding_idx, max_norm, norm_type, runtime_tuner):
        input, weight = ctx.saved_tensors

        # Sync parameters for backward computation
        sync_param(weight, rank_id=self.weight.rank_id)
        
        # Compute gradients
        if ctx.needs_input_grad[1]:
            grad_weight = ops.embedding_weight_grad(grad_output, input, weight, runtime_tuner)
            if self.weight.bwd_sync:
                sync_grad(grad_weight, async_op=False, rank_id=self.weight.rank_id)
        else:
            grad_weight = None

        # Check if the grad shape is correct
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")

        # Desync the grad for non-owner ranks
        if ctx.needs_input_grad[1] and hasattr(self.weight, 'bwd_sync') and self.weight.bwd_sync:
            grad_weight = desync_grad(grad_weight, rank_id=self.weight.rank_id)
            
        # Desync parameters after backward
        desync_param_data(weight, rank_id=self.weight.rank_id)

        return grad_weight

