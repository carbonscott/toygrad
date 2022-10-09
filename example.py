#!/usr/bin/env python
# -*- coding: utf-8 -*-

from toygrad.scalar import Scalar

import torch

from graphviz import Digraph

def draw_dot(root):
  dot = Digraph(format='pdf', graph_attr={'rankdir': 'LR'}, filename = 'test.gv') # LR = left to right

  nodes, edges = root.build_graph()
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | value %.4f | grad %.4f }" % (n.label, n.value, n.grad), shape='record')
    if n.operator:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n.operator, label = n.operator)
      # and connect this node to it
      dot.edge(uid + n.operator, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2.operator)

  return dot

a1 = Scalar(4.0 , 'a1')
a2 = Scalar(-3.0, 'a2')
a3 = Scalar(2.0 , 'a3')

res_a = a1 * a2 + a2 * a3 + a3
res_a.backward()
draw_dot(res_a).render()


b1 = torch.tensor(4.0 , requires_grad = True)
b2 = torch.tensor(-3.0, requires_grad = True)
b3 = torch.tensor(2.0 , requires_grad = True)

res_b = b1 * b2 + b2 * b3 + b3
res_b.backward()


assert a1.grad == b1.grad, "Test failed!"
assert a2.grad == b2.grad, "Test failed!"
assert a3.grad == b3.grad, "Test failed!"
