
import torch
import torch.nn as nn

from .module import (
    Linear,
    LayerNorm,
    Embedding,
)
from ..utils.wrapper import wrap_layers

class DDP(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        wrap_layers(model, 
                    _target_modules,
                    _supported_modules,
                    _get_init_args,
                    auto_tune=False)
        _error_handling(model)
        self.module = model
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        if self.require_backward_grad_sync:
            self.enable_grad_sync()
        return self.module(*args, **kwargs)

    def enable_grad_sync(self):
        for param in self.module.parameters():
            setattr(param, 'bwd_sync', True)


_target_modules = [
    nn.Linear,
    nn.LayerNorm,
    nn.Embedding,
]

_supported_modules = {
    nn.Linear: Linear,
    nn.LayerNorm: LayerNorm,
    nn.Embedding: Embedding,
}

def _get_init_args(module):
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
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

def _error_handling(model: nn.Module):
    for name, param in model.named_parameters():
        if not hasattr(param, "bwd_sync"):
            raise NotImplementedError(f"Module {name} is not supported yet.")
