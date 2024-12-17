from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        axes = 1
        max_z = array_api.max(Z, axis=axes, keepdims=True)
        log_sum_exp = max_z + array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=axes, keepdims=True))
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        y = exp(Z)
        grad = summation(out_grad, axes=1).reshape((Z.shape[0], 1))
        sum_y = summation(y, axes=1).reshape((Z.shape[0], 1))
        grad = grad / sum_y
        grad = broadcast_to(grad, Z.shape)
        grad = y * grad
        return out_grad - grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


# TODO: 公式的功能和原理还没有看懂，暂时忽略。后续学习。
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))
        max_z = Z.max(axis=self.axes, keepdims=True)
        var = array_api.log(array_api.sum(array_api.exp(Z - max_z.broadcast_to(Z.shape)), axis=self.axes))
        return var + max_z.reshape(var.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))

        Z = node.inputs[0]
        shape = [1 if i in self.axes else Z.shape[i] for i in range(len(Z.shape))]
        gradient = exp(Z - node.reshape(shape).broadcast_to(Z.shape))
        return out_grad.reshape(shape).broadcast_to(Z.shape) * gradient
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

