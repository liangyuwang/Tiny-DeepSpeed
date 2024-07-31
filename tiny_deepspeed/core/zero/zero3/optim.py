# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import torch
from collections import OrderedDict
import torch.distributed as dist

from ..utils.partition import partition_tensors
from ...optim import sgd, adamw


def sync_grads(grad, rank):      # communication complexity: g
    # TODO: make communication-compute overlap
    #   Finished, the grad sync is moved inside modules, and this sync_grads is deprecated.
    dist.reduce(grad, dst=rank)
    torch.cuda.synchronize()
    return grad

def gather_grads(param, rank):      # communication complexity: g
    # TODO: make communication-compute overlap
    dist.broadcast(param, src=rank)
    torch.cuda.synchronize()

def _step_fn(self):
    for name, param in self.parameters.items():
        rank = self.param_part_table[name]
        if rank == dist.get_rank() and param.grad is None:
            continue
        # param.grad = sync_grads(param.grad, rank)
        if rank == dist.get_rank():
            param = self.one_step(name, param)
        gather_grads(param, rank)
        self._zero_grad(param)


class SGD(sgd.SGD):
    def __init__(self, parameters, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False, 
                 param_part_table: OrderedDict=None, ranks_map: list=None):
        self.param_part_table = param_part_table
        self.ranks_map = ranks_map
        super().__init__(parameters, lr, momentum, dampening, weight_decay, nesterov, maximize)
    
    def _init_opt(self):
        # Initialize velocity for the local partition
        if self.momentum != 0:
            self.velocities = OrderedDict()
            if self.param_part_table == None:
                self.param_part_table = OrderedDict()
                # Fake init velocities to avoid init it on single device
                for name, param in self.parameters.items():
                    self.velocities[name] = torch.zeros_like(param, device="meta")
                self.param_part_table, self.velocities = partition_tensors(self.velocities, ranks_map=self.ranks_map, evenness_priority=0)
                # Actual init velocities
                for name, _ in self.velocities.items():
                    if dist.get_rank() == self.param_part_table[name]:
                        self.velocities[name] = torch.zeros_like(self.velocities[name], device=self.param_part_table[name])
            else:
                # Init velocities
                for name, param in self.parameters.items():
                    if dist.get_rank() == self.param_part_table[name]:
                        self.velocities[name] = torch.zeros_like(param, device=self.param_part_table[name])

    def step(self):
        _step_fn(self)
    

class AdamW(adamw.AdamW):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, 
                 param_part_table: OrderedDict=None, ranks_map: list=None):
        self.param_part_table = param_part_table
        self.ranks_map = ranks_map
        super().__init__(parameters, lr, betas, eps, weight_decay, amsgrad)

    def _init_opt(self):
        # Initialize velocity for the local partition
        self.moments = OrderedDict()
        self.velocities = OrderedDict()
        if self.amsgrad:
            self.max_squared = OrderedDict()
        if self.param_part_table == None:
            self.param_part_table = OrderedDict()
            # Fake init velocities to avoid init it on single device
            for name, param in self.parameters.items():
                self.moments[name] = torch.zeros_like(param, device="meta")
                self.velocities[name] = torch.zeros_like(param, device="meta")
                if self.amsgrad:
                    self.max_squared[name] = torch.zeros_like(param, device="meta")
            self.param_part_table, self.moments = partition_tensors(self.moments, ranks_map=self.ranks_map, evenness_priority=0)
            _, self.velocities = partition_tensors(self.velocities, ranks_map=self.ranks_map, evenness_priority=0)
            if self.amsgrad:
                _, self.max_squared = partition_tensors(self.max_squared, ranks_map=self.ranks_map, evenness_priority=0)
            # Actual init
            for name, _ in self.parameters.items():
                if dist.get_rank() == self.param_part_table[name]:
                    self.moments[name] = torch.zeros_like(self.moments[name], device=self.param_part_table[name])
                    self.velocities[name] = torch.zeros_like(self.velocities[name], device=self.param_part_table[name])
                    if self.amsgrad:
                        self.max_squared[name] = torch.zeros_like(self.max_squared[name], device=self.param_part_table[name])
        else:
            # Init
            for name, param in self.parameters.items():
                if dist.get_rank() == self.param_part_table[name]:
                    self.moments[name] = torch.zeros_like(param, device=self.param_part_table[name])
                    self.velocities[name] = torch.zeros_like(param, device=self.param_part_table[name])
                    if self.amsgrad:
                        self.max_squared[name] = torch.zeros_like(param, device=self.param_part_table[name])
        
    def step(self):
        _step_fn(self)
