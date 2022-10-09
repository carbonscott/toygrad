'''

The goal of the toy grad engine is to practice the concepts that I learned
from (`micrograd`)[https:
//github.com/karpathy/micrograd/blob/master/micrograd/engine.py].  More
specifically, I want to achieve:

- Write a Tensor class that tracks value, grad and graph.
- Provide backprop both in the partial gradient level and global level.
- Try to pass unit tests that use PyTorch as the engine.

'''

import numpy as np

class Tensor:

    def __init__(self, batch_value, batch_grad = 0.0):
        self.batch_value = np.array(batch_value)
        self.batch_grad  = np.array(batch_grad)

        self.batch_prev_list = np.array([])
        self._backward = None


    def __repr__(self):
        return f"Tensor(Value:{self.batch_value}, Grad:{self.batch_grad})"


    def __add__(self, batch_input):
        batch_input = batch_input if isinstance(batch_input, Tensor) else Tensor(batch_input)
        res = self.batch_value + batch_input.batch_value
        res = Tensor(res)

        def _backward():
            ''' 
            Backprop for addition:
                Basically just pass along the gradient from resulting node.
            '''
            self.batch_grad        += 1.0
            batch_input.batch_grad += 1.0

        self._backward = _backward

        return res


    def __mul__(self, batch_input):
        batch_input = batch_input if isinstance(batch_input, Tensor) else Tensor(batch_input)
        res = self.batch_value * batch_input.batch_value
        res = Tensor(res)

        def _backward():
            ''' 
            Backprop for multiplication:
                Basically just pass along the gradient from resulting node.
            '''
            self.batch_grad        += res.batch_grad
            batch_input.batch_grad += res.batch_grad

        self._backward = _backward

        return res


    ## def build_graph(self):
