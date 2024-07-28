# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch

from example.model import GPTConfig, GPT2Model
from tiny_deepspeed.core.optim import SGD, AdamW

# init distributed
torch.manual_seed(0)
torch.cuda.set_device(0)
device = torch.device("cuda:0")

config = GPTConfig()
input = torch.randint(0, config.vocab_size, (1, config.block_size)).to(device)
target = torch.randint(0, config.vocab_size, (1, config.block_size)).to(device)
model = GPT2Model(config).to(device)
optimizer = AdamW(model.named_parameters(), lr=1e-5, weight_decay=1e-1)

for i in range(100):
    _, loss = model(input, target)
    loss.backward()
    optimizer.step()
    print(f"iter {i} loss: {loss.item():.4f}")

