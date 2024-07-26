
import torch
import torch.distributed as dist

from . import sgd, adamw


def sync_grads(grad):
    # TODO: make communication-compute overlap
    dist.all_reduce(grad)  # communication complexity: 2g
    torch.cuda.synchronize()
    return grad

def _step_fn(self):
    for name, param in self.parameters.items():
        if param.grad is None:
            continue
        sync_grads(param.grad)
        param = self.one_step(name, param)
        self._zero_grad(param)


class SGD(sgd.SGD):
    def step(self):
        _step_fn(self)

class AdamW(adamw.AdamW):
    def step(self):
        _step_fn(self)
    