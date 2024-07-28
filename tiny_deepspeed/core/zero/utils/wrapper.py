# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn


def wrap_layers(
        model: nn.Module, 
        target_modules: list, 
        supported_modules: list,
        get_init_args: dict,
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
                child_device = next(child.parameters()).device
                new_module = new_module.to(child_device)
                new_module.load_state_dict(child.state_dict())
                new_module.train(child.training)
                setattr(model, child_name, new_module)
            elif not isinstance(child, tuple(target_modules)):
                _replace_module_recursive(child, full_name)
    _replace_module_recursive(model)
