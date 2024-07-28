# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
from typing import Tuple

from ...autotuner import RuntimeAutoTuner


def embedding_forward(input: torch.Tensor, weight: torch.Tensor, 
                      padding_idx: int, max_norm: float, norm_type: float,
                      runtime_tuner: RuntimeAutoTuner=None):
    if runtime_tuner:
        tuned_func = runtime_tuner.choose_function(
            [_embedding_forward_torch, ],   # Add more functions here
            input, weight, padding_idx, max_norm, norm_type
        )
        return tuned_func(input, weight, padding_idx, max_norm, norm_type)
    else:
        return _embedding_forward_torch(input, weight, padding_idx, max_norm, norm_type)

def embedding_weight_grad(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, runtime_tuner: RuntimeAutoTuner=None):
    if runtime_tuner:
        tuned_func = runtime_tuner.choose_function(
            [_embedding_weight_grad_torch, ],   # Add more functions here
            grad_output, input, weight
        )
        return tuned_func(grad_output, input, weight)
    else:
        return _embedding_weight_grad_torch(grad_output, input, weight)


def _embedding_forward_torch(input: torch.Tensor, weight: torch.Tensor, 
                             padding_idx: int, max_norm: float, norm_type: float):
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    input_size = list(input.size())
    output = weight.index_select(0, input.view(-1)).view((*input_size, weight.size(-1)))
    return output

def _embedding_weight_grad_torch(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
    grad_weight = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
    input_flat = input.view(-1)
    grad_output_flat = grad_output.view(-1, weight.size(1))
    grad_weight.index_add_(0, input_flat, grad_output_flat)
    return grad_weight

def _no_grad_embedding_renorm_(weight: torch.Tensor, input: torch.Tensor, max_norm: float, norm_type: float) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.embedding_renorm_(weight.detach(), input, max_norm, norm_type)
