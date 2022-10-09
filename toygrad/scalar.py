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

    def __init__(self, value):
        self.value = value
        self.grad  = 0.0

        self.prev_list = []
        self._backward = lambda: None    # Returns None from a parameter-less function by default


    def __repr__(self):
        return f"Scalar(Value:{self.value}, Grad:{self.grad})"


    def __add__(self, input):
        input = input if isinstance(input, Scalar) else Scalar(input)
        res = self.value + input.value

        res = Scalar(res)
        res.prev_list = (self, input)

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


    def backward(self):
        graph_as_list = []
        visited_list = set()

        def build_graph(node):
            # Have we been this node before???
            if not node in visited_list:
                # Note down that we have visited this node...
                visited_list.add(node)

                # Going down the tree until no more node is found before saving this node...
                # Implicit conditional branches in this recursion:
                # - Loop done scenario 1 (end)         : no more prev nodes exist;
                # - Loop done Scenario 2 (intermediate): no more prev nodes unvisited;
                for prev_node in node.prev_list:
                    build_graph(prev_node)

                # Save end/intermediate node...
                graph_as_list.append(node)

        build_graph(self)

        # Go back to the result node and calculate gradients...
        self.grad = 1.0
        for node in reversed(graph_as_list):
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
