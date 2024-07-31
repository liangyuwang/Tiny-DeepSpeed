# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn
from collections import OrderedDict

from .module import (
    Linear,
    LayerNorm,
    Embedding,
)
from ..utils.wrapper import wrap_layers, error_handling

class Zero2(nn.Module):
    def __init__(self, model: nn.Module, param_part_table: OrderedDict):
        super().__init__()
        wrap_layers(model, 
                    _supported_modules,
                    auto_tune=False)
        error_handling(model)
        self.module = model
        self.param_part_table = param_part_table
        self.require_backward_grad_sync = False
        self.set_rank_id()
    
    def forward(self, *args, **kwargs):
        if self.require_backward_grad_sync:
            self.enable_grad_sync()
        self.require_backward_grad_sync = False
        return self.module(*args, **kwargs)

    def set_rank_id(self):
        for name, param in self.module.named_parameters():
            rank_id = self.param_part_table[name]
            setattr(param, 'rank_id', rank_id)

    def enable_grad_sync(self):
        for param in self.module.parameters():
            setattr(param, 'bwd_sync', True)


_supported_modules = {
    nn.Linear: Linear,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
}
