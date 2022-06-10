import numpy as np

def identity(x):
  return x

def tanh(x):
  return np.tanh(x)

def d_tanh(x):
  return 1 - x ** 2

def relu(x):
  return np.maximum(x, 0)

def d_relu(x):
  return 1 if x > 0 else 0