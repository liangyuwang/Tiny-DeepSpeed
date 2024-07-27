# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0



from .sgd import SGD
from .adamw import AdamW

from .ddp import SGD as DDPSGD
from .ddp import AdamW as DDPAdamW

from .zero1 import SGD as Zero1SGD
from .zero1 import AdamW as Zero1AdamW