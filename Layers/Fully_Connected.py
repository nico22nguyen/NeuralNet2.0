import numpy as np

class Fully_Connected:
  def __init__(self, num_inputs: int, num_neurons: int):
    # gaussian distribution of weights and biases
    self.weights = np.random.rand(num_inputs, num_neurons)
    self.biases = np.random.rand(1, num_neurons)

    # used to simplify back-propagation
    self.input = None

  def forward_pass(self, _input: list):
    self.input = _input
    return _input @ self.weights + self.biases

  # returns new "running gradient"
  def back_propagate(self, running_gradient):
    # calculate gradients for weights and biases
    self.d_weights = self.input.T @ running_gradient
    self.d_biases = np.ones((1, running_gradient.shape[0])) @ running_gradient

    # tack this layer's weights onto running gradient and return
    return running_gradient @ self.weights.T
  
  def update(self):
    self.weights = self.weights + self.d_weights * 0.0000001
    self.biases = self.biases + self.d_biases * 0.0000001