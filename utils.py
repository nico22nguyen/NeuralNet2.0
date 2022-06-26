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
  return (np.maximum(x, 0) == x) * 1

def squared_error(A, B):
  return np.square(np.subtract(A,  B))

# TODO: np.exp and np.sum can be trivially combined into one loop as an optimization
def softmax(x):
  exponentiated = np.exp(x)
  sum = np.sum(exponentiated)

  return exponentiated / sum

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

def sigmoid(x):
  return 1 / (1 + np.exp(x))

def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))

def one_hot(value, total_values):
  return np.array([1 if i == value else 0 for i in range(total_values)])

def categorical_ce(labels, actual, epsilon=1e-5): 
  return labels * (-np.log(actual + epsilon)) + (1 - labels) * (-np.log(1 - actual + epsilon))

def d_categorical_ce(labels, actual, epsilon=1e-5):
  return -(labels / (actual + epsilon))
  
def mean_squared_error(labels, actual):
  return np.linalg.norm((labels - actual) ** 2)

def d_mean_squared_error(labels, actual):
  return 2 * (labels - actual)