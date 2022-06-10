import numpy as np
import utils

class Layer:
  def __init__(self, num_neurons, num_inputs, activation):
    self.weights = np.random.randn(num_neurons, num_inputs)
    self.biases = np.random.randn(num_neurons)
    self.current_inputs = []

    # default activation is none
    self.activation = utils.identity
    self.d_activation = utils.identity

    if (activation == 'tanh'):
      self.activation = utils.tanh
      self.d_activation = utils.d_tanh
    elif (activation == 'relu'):
      self.activation = utils.relu
      self.d_activation = utils.d_relu

  def forward_pass(self, inputs):
    self.current_inputs = inputs
    return self.activation(np.dot(self.weights, inputs) + self.biases)

  # returns new "running gradient"
  def back_propagate(self, running_gradient):
    # calculate gradient for activation function
    activation_gradient = self.d_activation(self.current_inputs)

    # tack activation gradient onto running gradient
    new_running_gradient = np.dot(running_gradient, activation_gradient)

    # calculate gradients for weights and biases
    d_weights = np.dot(new_running_gradient, self.current_inputs)
    d_biases = new_running_gradient

    # adjust weights and biases
    self.weights = np.subtract(self.weights, d_weights)
    self.biases = np.subtract(self.biases, d_biases)

    # tack this layer's weights onto running gradient and return
    return np.dot(new_running_gradient, self.weights)
