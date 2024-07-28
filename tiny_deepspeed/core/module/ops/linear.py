# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
from ...autotuner import RuntimeAutoTuner


def linear_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None, runtime_tuner: RuntimeAutoTuner=None):
    if runtime_tuner:
        tuned_func = runtime_tuner.choose_function(
            [_linear_forward_torch, ],   # Add more functions here
            input, weight, bias
        )
        return tuned_func(input, weight, bias)
    else:
        return _linear_forward_torch(input, weight, bias)

def linear_input_grad(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, runtime_tuner: RuntimeAutoTuner=None):
    if runtime_tuner:
        tuned_func = runtime_tuner.choose_function(
            [_linear_input_grad_torch, ],   # Add more functions here
            grad_output, input, weight
        )
        return tuned_func(grad_output, input, weight)
    else:
        return _linear_input_grad_torch(grad_output, input, weight)

def linear_weight_grad(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, runtime_tuner: RuntimeAutoTuner=None):
    if runtime_tuner:
        tuned_func = runtime_tuner.choose_function(
            [_linear_weight_grad_torch, ],   # Add more functions here
            grad_output, input, weight
        )
        return tuned_func(grad_output, input, weight)
    else:
        return _linear_weight_grad_torch(grad_output, input, weight)

def linear_bias_grad(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, runtime_tuner: RuntimeAutoTuner=None):
    if runtime_tuner:
        tuned_func = runtime_tuner.choose_function(
            [_linear_bias_grad_torch, ],   # Add more functions here
            grad_output, input, weight
        )
        return tuned_func(grad_output, input, weight)
    else:
        return _linear_bias_grad_torch(grad_output, input, weight)


def _linear_forward_torch(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor=None):
    output = input @ weight.t()
    if bias is not None:
        output += bias
    return output

def _linear_input_grad_torch(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
    return grad_output @ weight

def _linear_weight_grad_torch(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
    if input.dim() == 1:
        input = input.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    if input.dim() >= 3:
        B, M, N = input.shape[0], input.shape[-1], grad_output.shape[-1]
        K = input.numel() // (B * M)
        input = input.reshape(B * K, M)
        grad_output = grad_output.reshape(B * K, N)
    return grad_output.t() @ input

def _linear_bias_grad_torch(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
    if input.dim() > 3:
        B, M, N = input.shape[0], input.shape[-1], grad_output.shape[-1]
        K = input.numel() // (B * M)
        grad_output = grad_output.reshape(B * K, N)
    return grad_output.sum(0)
