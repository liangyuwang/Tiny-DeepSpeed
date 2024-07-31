# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
from torch.nn.modules.normalization import (
    _shape_t,
    numbers
)

from .ops.layernorm import (
    layernorm_fwd,
    layernorm_dx,
    layernorm_dwdb
)
from ..autotuner import RuntimeAutoTuner

class LayerNorm(nn.LayerNorm):
    
    def __init__(
        self, 
        normalized_shape: _shape_t, 
        eps: float = 1e-5, 
        elementwise_affine: bool = True,
        bias: bool = True, 
        device=None, 
        dtype=None,
        auto_tune: bool = False
    ) -> None:
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.use_bias = False if bias is None else True
        if not self.use_bias:
            raise NotImplementedError("Currently bias must be enabled. Set use_fused=False to use the default implementation.")
        if not elementwise_affine:
            raise NotImplementedError("Currently elementwise_affine must be True.")
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps, elementwise_affine, self.use_bias, device, dtype)
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.runtime_tuner = RuntimeAutoTuner(enable=auto_tune) if auto_tune else None
        self._init_parameters()
    
    def _init_parameters(self):
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
            if self.use_bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, **self.factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.runtime_tuner:
            self.runtime_tuner.final_tune()
        if len(self.normalized_shape) != 1 or self.normalized_shape[0] != input.shape[-1]:
            raise NotImplementedError("Currently layernorm only support normalized_dim=-1.")
        output = ApplyLayernormFunc(self.eps, self.runtime_tuner, 
                                    self.forward_callback,
                                    self.backward_callback)(
                                        input, self.weight, self.bias)
        return output

    def forward_callback(self, ctx, input, weight, bias, eps, runtime_tuner):
        output, mean, rstd, args = layernorm_fwd(input, weight, bias, eps, runtime_tuner)
        ctx.save_for_backward(input, weight, bias, mean, rstd)
        ctx.args = args
        return ctx, output

    def backward_callback(self, ctx, grad_output, eps, runtime_tuner):
        input, weight, bias, mean, rstd = ctx.saved_tensors
        args = {
            'BLOCK_SIZE': ctx.args['BLOCK_SIZE'],
            'num_warps': ctx.args['num_warps'],
            'eps': eps,
        }
        dx, dw_, db_, args = layernorm_dx(grad_output, input, weight, bias, mean, rstd, args, runtime_tuner)
        dw, db = layernorm_dwdb(weight, bias, dw_, db_, args, runtime_tuner)
        
        # Check if the grad shape is correct
        if dx is not None and dx.shape != input.shape:
            raise RuntimeError(f"grad_input shape {dx.shape} is not equal to input shape {input.shape}")
        if dw is not None and dw.shape != weight.shape:
            raise RuntimeError(f"grad_weight shape {dw.shape} is not equal to weight shape {weight.shape}")
        if db is not None and db.shape != bias.shape:
            raise RuntimeError(f"grad_bias shape {db.shape} is not equal to bias shape {bias.shape}")

        return dx, dw, db



def ApplyLayernormFunc(eps, runtime_tuner, forward_callback, backward_callback):
    class FusedFunction(torch.autograd.function.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx, output = forward_callback(ctx, input, weight, bias, eps, runtime_tuner)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            dx, dw, db = backward_callback(ctx, grad_output, eps, runtime_tuner)
            return dx, dw, db
    return FusedFunction.apply
