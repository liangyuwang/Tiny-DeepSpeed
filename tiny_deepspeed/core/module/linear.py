# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn

from .ops.linear import (
    linear_forward, 
    linear_input_grad, 
    linear_weight_grad, 
    linear_bias_grad
)
from ..autotuner import RuntimeAutoTuner

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, auto_tune: bool = False):
        self.use_bias = False if bias is None else True
        super().__init__(in_features, out_features, self.use_bias)
        self.in_features = in_features
        self.out_features = out_features
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.runtime_tuner = RuntimeAutoTuner(enable=auto_tune) if auto_tune else None
        self._init_parameters()
    
    def _init_parameters(self):
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.runtime_tuner:
            self.runtime_tuner.final_tune()
        output = _ApplyLinearFunc(self.runtime_tuner, 
                                  self.forward_callback,
                                  self.backward_callback)(
                                    input, self.weight, self.bias)
        return output
    
    def forward_callback(self, ctx, input, weight, bias, runtime_tuner):
        ctx.save_for_backward(input, weight, bias)
        output = linear_forward(input, weight, bias, runtime_tuner)
        return ctx, output

    def backward_callback(self, ctx, grad_output, runtime_tuner):
        input, weight, bias = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = linear_input_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            grad_weight = linear_weight_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_weight = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = linear_bias_grad(grad_output, input, weight, runtime_tuner)
        else:
            grad_bias = None
        
        # Check if the grad shape is correct
        if grad_input is not None and grad_input.shape != input.shape:
            raise RuntimeError(f"grad_input shape {grad_input.shape} is not equal to input shape {input.shape}")
        if grad_weight is not None and grad_weight.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {grad_weight.shape} is not equal to weight shape {weight.shape}")
        if grad_bias is not None and grad_bias.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {grad_bias.shape} is not equal to bias shape {bias.shape}")
        
        return grad_input, grad_weight, grad_bias



def _ApplyLinearFunc(runtime_tuner, forward_callback, backward_callback):
    """
        Returns a function that computes the linear function.
    """
    class LinearFunc(torch.autograd.function.Function):
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            ctx, output = forward_callback(ctx, input, weight, bias, runtime_tuner)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            return backward_callback(ctx, grad_output, runtime_tuner)
    return LinearFunc.apply
