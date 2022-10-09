#!/usr/bin/env python
# -*- coding: utf-8 -*-

from toygrad.scalar import Scalar

import torch

a1 = Scalar(2.0)
a2 = Scalar(3.0)

t1 = a1 * a2
t2 = a1 * a1
t3 = t1 + t2
t4 = t3 + a2

t4.backward()

b1 = torch.tensor(2.0, requires_grad = True)
b2 = torch.tensor(3.0, requires_grad = True)

s1 = b1 * b2
s2 = b1 * b1
s3 = s1 + s2

s2.retain_grad()
s3.retain_grad()

s4 = s3 + b2

s4.backward()
