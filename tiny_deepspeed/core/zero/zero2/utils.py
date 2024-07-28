
import torch.nn as nn

class Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, bwd_sync=True):
        t = nn.Parameter.__new__(cls, data, requires_grad)
        t.bwd_sync = bwd_sync
        return t

