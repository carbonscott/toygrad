'''

The goal of the toy grad engine is to practice the concepts that I learned
from (`micrograd`)[https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py].  
More specifically, I want to achieve:

- Write a Scalar class that tracks value, grad and prev nodes that lead to the
  current node (self).
- Provide backprop both in the partial gradient level and global level.
- Try to pass unit tests that use PyTorch as the engine.

Some extra points to understand:

- Multiply a value less than one (like 1e-3) can cause the gradient to be very
  small.
- Adding a value that has been used before in some multiplication can make its
  graident increased by one.

'''

class Scalar:
    '''
    Property of accumulative gradient of an operand:
        Gradient of the responding result from an operator with respect to each
        operand is accumulative.  

    Example:
        res = a1 * a2 + a2

        d(res)/d(a2) = p(res)/p(a1 * a2) * p(a1 * a2)/p(a2) +
                       p(res)/p(a2)
    '''

    def __init__(self, value, label = ''):
        self.value = value
        self.grad  = 0.0

        self.operand_list = []
        self.operator     = ''
        self._backward    = lambda: None    # Returns None from a parameter-less function by default

        self.label = label


    def __repr__(self):
        return f"Scalar(Value:{self.value}, Grad:{self.grad}), Label:{self.label}"


    def __add__(self, input):
        input = input if isinstance(input, Scalar) else Scalar(input)
        res = self.value + input.value

        # Track the result, its operands, operator and the label...
        res = Scalar(res)
        res.operand_list = (self, input)
        res.operator = '+'
        res.label = f'({self.label} + {input.label})'

        def _backward():
            # Accumulate gradients for each operand...
            self.grad  += res.grad
            input.grad += res.grad

        # The resulting node should have a definite backward function...
        res._backward = _backward

        return res


    def __mul__(self, input):
        input = input if isinstance(input, Scalar) else Scalar(input)
        res = self.value * input.value

        res = Scalar(res)
        res.operand_list = (self, input)
        res.operator = '*'
        res.label = f'({self.label} * {input.label})'

        def _backward():
            # Accumulate gradients for each operand scaled by the value of the other operand (=value of partial gradient)...
            grad_partial = input.value
            self.grad   += res.grad * grad_partial

            grad_partial = self.value
            input.grad  += res.grad * grad_partial

        # The resulting node should have a definite backward function...
        res._backward = _backward

        return res


    def build_graph(self):
        # Set up global variables accessible through the following closure functions...
        node_list = []
        edge_list = []

        # Define the closure...
        def trace(node_result):
            ''' The function is writte in a closure style so that all recursive
                instances can access to the same global variable (node_list and
                edge_list).
            '''
            # Going down the tree until no more node is found before saving this node...
            # Implicit conditional branches in this recursion:
            # - End scenario 1 (end)         : no more prev nodes exist;
            # - End scenario 2 (intermediate): no more prev nodes unvisited;
            for node_operand in node_result.operand_list:
                trace(node_operand)
                edge_list.append((node_operand, node_result))

            # Save end/intermediate node...
            node_list.append(node_result)

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
