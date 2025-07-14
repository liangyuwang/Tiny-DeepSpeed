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
from ..utils.wrapper import target_modules, error_handling, get_init_args

class Zero3(nn.Module):
    def __init__(self, model: nn.Module, param_part_table: OrderedDict):
        super().__init__()
        self.module = model
        self.param_part_table = param_part_table
        self.require_backward_grad_sync = False
        self.wrap_layers(self.module, 
                    _supported_modules,
                    auto_tune=False)
        error_handling(self.module)
        self.set_param_info()
        self.reinit()
        self.to_rank()
        self.set_param_info()
        self.enable_grad_sync()  # Enable grad sync by default
    
    def forward(self, *args, **kwargs):
        # if self.require_backward_grad_sync:
        #     self.enable_grad_sync()
        # self.require_backward_grad_sync = False
        return self.module(*args, **kwargs)
    
    def reinit(self):
        def _replace_module_recursive(model, path=''):
            for child_name, child in model.named_children():
                full_name = f"{path}.{child_name}" if path else child_name
                if hasattr(child, "reinit_parameters"):
                    child.reinit_parameters()
                else:
                    _replace_module_recursive(child, full_name)
        _replace_module_recursive(self.module)
    
    def to_rank(self):
        for name, param in self.module.named_parameters():
            rank_id = self.param_part_table[name]["rank_id"]
            device = f"cuda:{rank_id}"
            param.to(device)

    def set_param_info(self):
        for name, param in self.module.named_parameters():
            if isinstance(self.param_part_table[name], dict):
                info = self.param_part_table[name]
                rank_id = info["rank_id"]
                size = info["cached_size"]
                dtype = info["cached_dtype"]
                setattr(param, 'rank_id', rank_id)
                setattr(param, 'cached_size', size)
                setattr(param, 'cached_dtype', dtype)
            else:
                rank_id = self.param_part_table[name]
                setattr(param, 'rank_id', rank_id)
                size = tuple(param.size())
                dtype = param.dtype
                self.param_part_table[name] = {"rank_id": rank_id,
                                               "cached_size": size,
                                               "cached_dtype": dtype}

    def enable_grad_sync(self):
        for param in self.module.parameters():
            setattr(param, 'bwd_sync', True)
            setattr(param, 'fwd_sync', True)

    def wrap_layers(
            self,
            model: nn.Module, 
            supported_modules: list,
            auto_tune: bool=False
        ):
        """
        Wrap the selected layers with appropriate modules based on the specified criteria.
        
        Args:
            model (nn.Module): The original model to modify.
            target_modules (list): List of module types to be replaced.
            auto_tune (bool): If using auto tuning or not.
        """
        def _replace_module_recursive(model, path=''):
            for child_name, child in model.named_children():
                full_name = f"{path}.{child_name}" if path else child_name
                if isinstance(child, tuple(target_modules)):
                    module_class = supported_modules[type(child)]
                    child_init_args = get_init_args(child)
                    new_module = module_class(**child_init_args, auto_tune=auto_tune)
                    new_module.load_state_dict(child.state_dict())
                    new_module.train(child.training)
                    setattr(model, child_name, new_module)
                elif not isinstance(child, tuple(target_modules)):
                    _replace_module_recursive(child, full_name)
        _replace_module_recursive(model)


_supported_modules = {
    nn.Linear: Linear,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
}
