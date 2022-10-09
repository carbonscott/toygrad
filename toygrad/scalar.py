'''

The goal of the toy grad engine is to practice the concepts that I learned
from (`micrograd`)[https:
//github.com/karpathy/micrograd/blob/master/micrograd/engine.py].  More
specifically, I want to achieve:

- Write a Scalar class that tracks value, grad and prev nodes that lead to the current node (self).
- Provide backprop both in the partial gradient level and global level.
- Try to pass unit tests that use PyTorch as the engine.

'''

class Scalar:

    def __init__(self, value, label = ''):
        self.value = value
        self.grad  = 0.0

        self.prev_list = []
        self._backward = lambda: None    # Returns None from a parameter-less function by default

        self._op = ''
        self.label = label


    def __repr__(self):
        return f"Scalar(Value:{self.value}, Grad:{self.grad}), Label:{self.label}"


    def __add__(self, input):
        input = input if isinstance(input, Scalar) else Scalar(input)
        res = self.value + input.value

        res = Scalar(res)
        res.prev_list = (self, input)
        res._op = '+'
        res.label = f'({self.label} + {input.label})'

        def _backward():
            ''' 
            Backprop for addition:
                Track each pathway.
            '''
            self.grad  += res.grad
            input.grad += res.grad

        # The resulting node should have a definite backward function...
        res._backward = _backward

        return res


    def __mul__(self, input):
        input = input if isinstance(input, Scalar) else Scalar(input)
        res = self.value * input.value

        res = Scalar(res)
        res.prev_list = (self, input)
        res._op = '*'
        res.label = f'({self.label} * {input.label})'

        def _backward():
            ''' 
            Backprop for multiplication:
                Basically just pass along the gradient from resulting node.
            '''
            par_grad = input.value
            self.grad  += res.grad * par_grad

            par_grad = self.value
            input.grad += res.grad * par_grad

        # The resulting node should have a definite backward function...
        res._backward = _backward

        return res


    def build_graph(self):
        # Set up global variables accessible through the following closure functions...
        node_list = []
        edge_list = []

        # Define the closure...
        def trace(node_current):
            ''' The function is writte in a closure style so that all recursive
                instances can access to the same global variable (node_list and
                edge_list).
            '''
            # Are we at the end node???
            if not node_current in node_list:
                # Going down the tree until no more node is found before saving this node...
                # Implicit conditional branches in this recursion:
                # - End scenario 1 (end)         : no more prev nodes exist;
                # - End scenario 2 (intermediate): no more prev nodes unvisited;
                for node_prev in node_current.prev_list:
                    trace(node_prev)
                    edge_list.append((node_prev, node_current))

                # Save end/intermediate node...
                node_list.append(node_current)

        # Trace down the graph from self...
        trace(self)

        return node_list, edge_list


    def backward(self):
        node_list, edge_list = self.build_graph()

        # Go back to the result node and calculate gradients...
        self.grad = 1.0
        for node in reversed(node_list):
            # Calculate gradients or do nothing if no prev node exists...
            node._backward()


    def __radd__(self, input):
        return self + input


    def __rmul__(self, input):
        return self * input


    def __neg__(self):
        return self * (-1)


    def __sub__(self, input):
        return self + (-input)


    def __rsub__(self, input):
        return input - self
