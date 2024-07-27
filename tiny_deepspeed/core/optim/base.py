# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0


from collections import OrderedDict

class Optimizer():
    def __init__(self, parameters):
        self.parameters = OrderedDict(parameters)
        self._init_opt()
    
    def _init_opt(self):
        pass

    def step(self):
        for name, param in self.parameters.items():
            if param.grad is None:
                continue
            param = self.one_step(name, param)
            self._zero_grad(param)
    
    def one_step(self, name, param):
        pass
    
    def _zero_grad(self, param):
        param.grad = None