# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
from collections import OrderedDict

from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False, maximize=False):
        if momentum < 0 or dampening < 0 or weight_decay < 0:
            raise ValueError("Momentum, dampening, and weight decay should be non-negative")
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        super().__init__(parameters)
    
    def _init_opt(self):
        if self.momentum != 0:
            # Initialize velocity for each parameter
            self.velocities = OrderedDict({k: torch.zeros_like(p, device=p.device) for k, p in self.parameters.items()})
    
    def one_step(self, name, param):
        grad = param.grad
        if self.weight_decay != 0:
            grad.add_(param.data, alpha=self.weight_decay)

        if self.maximize:
            grad = -grad

        if self.momentum != 0:
            v = self.velocities[name]
            v.mul_(self.momentum).add_(grad, alpha=1 - self.dampening)
            
            if self.nesterov:
                grad.add_(v, alpha=self.momentum)
            else:
                grad = v

        param.data.add_(grad, alpha=-self.lr)
        return param
    