# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torchvision.models import vit_b_16
from collections import OrderedDict

from tiny_deepspeed.core.optim import SGD, AdamW

# init distributed
torch.manual_seed(0)
torch.cuda.set_device(0)
device = torch.device("cuda:0")

input = torch.randn(1, 3, 224, 224).to(device)
target = torch.randint(0, 1000, (1,)).to(device)
model = vit_b_16().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.named_parameters(), lr=1e-5, weight_decay=1e-1)

for i in range(100):
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    print(f"iter {i} loss: {loss.item():.4f}")

