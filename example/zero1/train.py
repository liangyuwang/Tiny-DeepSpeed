# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.distributed as dist
from torchvision.models import vit_b_16
from collections import OrderedDict

from tiny_deepspeed.core.optim import Zero1SGD, Zero1AdamW
from tiny_deepspeed.core.utils.partition import partition_tensors

# init distributed
rank = int(os.getenv('LOCAL_RANK', '0'))
torch.manual_seed(rank)
world_size = int(os.getenv('WORLD_SIZE', '1'))
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.cuda.set_device(rank)

ranks_map = [f"cuda:{i}" for i in range(world_size)]
with torch.device('meta'):
    model = vit_b_16()
    parts, _ = partition_tensors(OrderedDict(model.named_parameters()), 
                                    ranks_map=ranks_map,
                                    evenness_priority=0,
                                    verbose=True)

input = torch.randn(1, 3, 224, 224).to(rank)
target = torch.randint(0, 1000, (1,)).to(rank)
model = vit_b_16().to(rank)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Zero1AdamW(model.named_parameters(), lr=1e-5, weight_decay=1e-1, param_part_table=parts, ranks_map=ranks_map)

for i in range(100):
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    if rank==0: print(f"iter {i} loss: {loss.item():.4f}")

dist.destroy_process_group()
