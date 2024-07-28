
from .optim import SGD, AdamW
from .zero.ddp import (
    DDPSGD, DDPAdamW,
    DDP
)
from .zero.zero1 import (
    Zero1SGD, Zero1AdamW
)

from .zero.utils.partition import partition_tensors