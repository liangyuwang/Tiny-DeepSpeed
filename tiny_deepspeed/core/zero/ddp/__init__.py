# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


from .optim import SGD as DDPSGD
from .optim import AdamW as DDPAdamW

from .wrapper import DDP