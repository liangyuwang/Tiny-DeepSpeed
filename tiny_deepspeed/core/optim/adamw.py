
import torch
from collections import OrderedDict

from .base import Optimizer

class AdamW(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("Learning rate, epsilon, and weight decay should be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Beta parameters should be in the range [0, 1)")
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        super().__init__(parameters)
    

    def _init_opt(self):
        self.moments = OrderedDict({k: torch.zeros_like(p) for k, p in self.parameters.items()})
        self.velocities = OrderedDict({k: torch.zeros_like(p) for k, p in self.parameters.items()})
        if self.amsgrad:
            self.max_squared = OrderedDict({k: torch.zeros_like(p) for k, p in self.parameters.items()})

    
    def one_step(self, name, param):
        grad = param.grad
        if self.weight_decay != 0:
            grad = grad.add(param.data, alpha=self.weight_decay)
        
        m = self.moments[name]
        m.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])

        v = self.velocities[name]
        v.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])

        if self.amsgrad:
            max_v = self.max_squared[name]
            torch.max(max_v, v, out=max_v)
            denom = max_v.sqrt().add_(self.eps)
        else:
            denom = v.sqrt().add_(self.eps)

        step_size = self.lr / denom
        param.data.addcdiv_(m, step_size, value=-1)
        return param