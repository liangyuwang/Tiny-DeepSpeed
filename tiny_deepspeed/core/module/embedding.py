# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
from typing import Optional

from .ops.embedding import (
    embedding_forward, 
    embedding_weight_grad
)
from ..autotuner import RuntimeAutoTuner

class Embedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[torch.Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None, auto_tune: bool = False) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self._weight = _weight
        self._freeze = _freeze
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparse = sparse
        self.runtime_tuner = RuntimeAutoTuner(enable=auto_tune) if auto_tune else None
        self._init_parameters()
    
    def _init_parameters(self):
        if self._weight is None:
            self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **self.factory_kwargs),
                                    requires_grad=not self._freeze)
            self.reset_parameters()
        else:
            assert list(self._weight.shape) == [self.num_embeddings, self.embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(self._weight, requires_grad=not self._freeze)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.runtime_tuner:
            self.runtime_tuner.final_tune()
        output = _ApplyEmbeddingFunc(self.padding_idx, self.max_norm, self.norm_type,
                                     self.runtime_tuner, 
                                     self.forward_callback, 
                                     self.backward_callback)(
                                         input, self.weight)
        return output
    
    def forward_callback(self, ctx, input, weight, padding_idx, max_norm, norm_type, runtime_tuner):
        ctx.save_for_backward(input, weight)
        output = embedding_forward(input, weight, padding_idx, max_norm, norm_type, runtime_tuner)
        return ctx, output

    def backward_callback(self, ctx, grad_output, padding_idx, max_norm, norm_type, runtime_tuner):
        input, weight = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_weight = embedding_weight_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_weight = None

        # Check if the grad shape is correct
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")

        return grad_weight



def _ApplyEmbeddingFunc(padding_idx, max_norm, norm_type, runtime_tuner, forward_callback, backward_callback):
    """
        Returns a function that computes the Embedding function.
    """
    class EmbeddingFunc(torch.autograd.function.Function):
        @staticmethod
        def forward(ctx, input, weight=None):
            ctx, output = forward_callback(ctx, input, weight, padding_idx, max_norm, norm_type, runtime_tuner)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            grad_weight = backward_callback(ctx, grad_output, padding_idx, max_norm, norm_type, runtime_tuner)
            return None, grad_weight
    return EmbeddingFunc.apply

