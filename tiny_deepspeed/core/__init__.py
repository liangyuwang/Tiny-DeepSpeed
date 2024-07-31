# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


from .optim import SGD, AdamW
from .zero.ddp import (
    DDPSGD, DDPAdamW,
    DDP
)
from .zero.zero1 import (
    Zero1SGD, Zero1AdamW,
    Zero1
)
from .zero.zero2 import (
    Zero2SGD, Zero2AdamW,
    Zero2
)
from .zero.zero3 import (
    Zero3SGD, Zero3AdamW,
    Zero3
)

from .zero.utils.partition import partition_tensors