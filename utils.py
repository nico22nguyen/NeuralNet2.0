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

def mean_squared_error(A, B):
  return np.square(np.subtract(A,  B)).mean()

# TODO: np.exp and np.sum can be trivially combined into one loop as an optimization
def softmax(x):
  exponentiated = np.exp(x)
  sum = np.sum(exponentiated)

  return [exp / sum for exp in exponentiated]

def d_softmax(x):
  update_vectors = []
  for i in range(len(x)):
    updates_from_this_neuron = []
    for j in range(len(x)):
      # softmax magic
      update_val = x[i] * (1 - x[j]) if i == j else -x[i] * x[j]
      updates_from_this_neuron.append(update_val)
    update_vectors.append(updates_from_this_neuron)

  # sum all the updates and return the derivative vector
  return np.sum(update_vectors, axis=0)

def one_hot(value, total_values):
  return [1 if i == value else 0 for i in range(total_values)]