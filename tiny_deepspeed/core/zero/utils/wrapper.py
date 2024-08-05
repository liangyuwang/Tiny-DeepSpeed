# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
import torch.nn as nn


def wrap_layers(
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
                child_device = next(child.parameters()).device
                new_module = new_module.to(child_device)
                new_module.load_state_dict(child.state_dict())
                new_module.train(child.training)
                setattr(model, child_name, new_module)
            elif not isinstance(child, tuple(target_modules)):
                _replace_module_recursive(child, full_name)
    _replace_module_recursive(model)



target_modules = [
    nn.Linear,
    nn.LayerNorm,
    nn.Embedding,
]

def get_init_args(module):
    """
    Extract initialization arguments from a module to properly initialize its counterpart.
    
    Args:
        module (nn.Module): Module from which to extract initialization parameters.
        
    Returns:
        dict: Initialization arguments needed for the module.
    """
    if isinstance(module, nn.Linear):
        return {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias
        }
    elif isinstance(module, nn.LayerNorm):
        return {
            "normalized_shape": module.normalized_shape,
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine,
            "bias": module.bias
        }
    elif isinstance(module, nn.Embedding):
        return {
            "num_embeddings": module.num_embeddings, 
            "embedding_dim": module.embedding_dim, 
            "padding_idx": module.padding_idx,
            "max_norm": module.max_norm, 
            "norm_type": module.norm_type, 
            "scale_grad_by_freq": module.scale_grad_by_freq,
            "sparse": module.sparse,
        }
    else:
        raise NotImplementedError(f"Unsupported module type: {type(module)}. Currently support modules: [nn.Linear, nn.LayerNorm, nn.Embedding].")

def error_handling(model: nn.Module):
    for name, param in model.named_parameters():
        if not hasattr(param, "bwd_sync"):
            raise NotImplementedError(f"Module {name} is not supported yet. Currently support modules: [nn.Linear, nn.LayerNorm, nn.Embedding].")

