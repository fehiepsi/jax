from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import jax.numpy as np
from jax.util import unzip2
from jax import jit, grad, lax, random

# scan = lax.scan


# scan :: (a -> b -> a) -> a -> [b] -> [a]

def scan(f, x, ys):
  xs = []
  for y in ys:
    x = f(x, y)
    xs.append(x)
  return xs

nin = 2
nhid = 3
nout = 4

bot = np.zeros(nhid)  # really should be 'bottom'

def rnn_step(params, hy, x):
  W, b = params
  h, _ = hy
  hx = np.concatenate([h, x])
  hy = np.tanh(np.dot(W, hx) + b)
  h, y = hy[:nhid], hy[nhid:]
  return h, y

def rnn_predict(params, xs):
  h_init = np.zeros(nhid)
  step = partial(rnn_step, params)
  _, ys = unzip2(scan(step, (h_init, bot), xs))
  return ys

W = np.zeros((nhid + nout, nhid + nin))
b = np.zeros(nhid + nout)
params = (W, b)

xs = np.zeros((10, nin))
ys = rnn_predict(params, xs)


###

# scan2 :: (a -> b -> (a, c)) -> a -> [b] -> ([a], [c])
# where pair really means pytree

def scan2(f, x, ys):
  xs = []
  zs = []
  for y in ys:
    x, z = f(x, y)
    xs.append(x)
    zs.append(z)
  return xs, zs
# that's only for pairs but in general we need pytrees. to know the output
# pytree structure, we'll play our usual games when we trace f


nin = 2
nhid = 3
nout = 4

def rnn_step(params, h, x):
  W, b = params
  hx = np.concatenate([h, x])
  hy = np.tanh(np.dot(W, hx) + b)
  h, y = hy[:nhid], hy[nhid:]
  return h, y

def rnn_predict(params, xs):
  h_init = np.zeros(nhid)
  step = partial(rnn_step, params)
  _, ys = scan2(step, h_init, xs)
  return ys

W = np.zeros((nhid + nout, nhid + nin))
b = np.zeros(nhid + nout)
params = (W, b)

xs = np.zeros((10, nin))
ys = rnn_predict(params, xs)

# scan2 ran on the first try
