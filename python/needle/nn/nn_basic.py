"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = Parameter(self.bias.reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # print(f"Linear forward: X={X}, weight={self.weight}")
        if X is None:
            print("Error: 'X' is None in Linear forward")
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias:
            y += self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        if len(X.shape) > 2:
            tmp=1
            for i in range(1, len(X.shape)):
                tmp*=X.shape[i]
            return X.reshape((X.shape[0], tmp))
        else:
            return X
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        one_hot = init.one_hot(num_classes, y)
        true_logits = ops.summation(logits * one_hot, axes=(1,))
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        loss = ops.summation(log_sum_exp-true_logits, axes=(0,))
        return loss/batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, layer_size = x.shape
        weight_broadcast = self.weight.broadcast_to((batch_size, layer_size))
        bias_broadcast = self.bias.broadcast_to((batch_size, layer_size))
        if self.training:
            batch_mean = (x.sum(axes=(0,))/batch_size)
            self.running_mean = (self.running_mean * (1 - self.momentum) + batch_mean * self.momentum).detach()
            batch_mean = batch_mean.reshape((1, layer_size)).broadcast_to((batch_size, layer_size))
            batch_var = (((x-batch_mean)**2).sum(axes=(0,))/batch_size)
            self.running_var = (self.running_var * (1 - self.momentum) + batch_var * self.momentum).detach()
            batch_var = batch_var.reshape((1, layer_size)).broadcast_to((batch_size, layer_size))
            batch_std = (batch_var + self.eps)**0.5
            return weight_broadcast * (x - batch_mean) / batch_std + bias_broadcast
        else:
            std_x = (x - self.running_mean.broadcast_to(x.shape)) / (self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5
            return weight_broadcast * std_x + bias_broadcast
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, layer_size = x.shape
        layer_mean = (x.sum(axes=(1,))/layer_size).reshape((batch_size, 1)).broadcast_to((batch_size, layer_size))
        layer_var = (((x-layer_mean)**2).sum(axes=(1,))/layer_size).reshape((batch_size, 1)).broadcast_to((batch_size, layer_size))
        layer_std = (layer_var + self.eps)**0.5
        weight_broadcast = self.weight.broadcast_to((batch_size, layer_size))
        bias_broadcast = self.bias.broadcast_to((batch_size, layer_size))
        y = weight_broadcast * (x-layer_mean) / layer_std + bias_broadcast
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_matrix = init.randb(*x.shape, p=(1-self.p))
            return x * drop_matrix / (1-self.p)
        else:
            # 漏加了这一句，debug了一下午。原因还没理解。
            # 明白原因了，因为没有加上下面这个逻辑，导致返回结果为None，这个问题其实可以通过模型结构一层一层推理出来。学习了。
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
