#!/usr/bin/env python
# -*- coding: utf-8 -*-

from toygrad.scalar import Scalar

import torch

from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = list(), list()
  def build(v):
    if v not in nodes:
      ## nodes.add(v)
      nodes.append(v)
      for child in v.prev_list:
        ## edges.add((child, v))
        edges.append((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='pdf', graph_attr={'rankdir': 'LR'}, filename = 'test.gv') # LR = left to right

  ## nodes, edges = trace(root)
  nodes, edges = root.build_graph()
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | value %.4f | grad %.4f }" % (n.label, n.value, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

a1 = Scalar(4.0, 'a1')
a2 = Scalar(-3.0, 'a2')
a3 = Scalar(2.0, 'a3')

res = a1 * a2 + a2 * a3
res.backward()
draw_dot(res).render()
