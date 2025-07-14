# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from collections import OrderedDict

from example.model import GPTConfigs, GPT2Model
from tiny_deepspeed.core import DDPSGD, DDPAdamW, DDP

# init distributed
rank = int(os.getenv('LOCAL_RANK', '0'))
torch.manual_seed(rank)
world_size = int(os.getenv('WORLD_SIZE', '1'))
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(rank)

config = GPTConfigs.gpt2
input = torch.randint(0, config.vocab_size, (1, config.block_size)).to(rank)
target = torch.randint(0, config.vocab_size, (1, config.block_size)).to(rank)
model = GPT2Model(config).to(rank)
model = DDP(model)
optimizer = DDPAdamW(model.named_parameters(), lr=1e-5, weight_decay=1e-1)

for i in tqdm(range(100), disable=rank!=0):
    model.require_backward_grad_sync = True # set to True when need grad all reduce
    _, loss = model(input, target)
    loss.backward()
    optimizer.step()
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    if rank==0: tqdm.write(f"iter {i} loss: {loss.item():.4f}")

dist.destroy_process_group()
