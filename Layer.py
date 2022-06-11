import numpy as np
import utils

class Layer:
  def __init__(self, num_neurons: int, num_inputs: int, activation: str):
    # gaussian distribution of weights and biases
    self.weights = np.random.randn(num_neurons, num_inputs)
    self.biases = np.random.randn(num_neurons)

    # used to simplify back-propagation
    self.current_input = None
    self.pre_activation_result = None

    # determine activation function
    self.activation, self.d_activation = get_activations_by_name(activation)

  def forward_pass(self, _input: list):
    self.current_input = _input
    self.pre_activation_result = np.dot(self.weights, _input) + self.biases

    return self.activation(self.pre_activation_result)

  # returns new "running gradient"
  def back_propagate(self, running_gradient):
    # calculate gradient for activation function
    activation_gradient = self.d_activation(self.pre_activation_result)

    # tack activation gradient onto running gradient
    new_running_gradient = np.dot(running_gradient, activation_gradient)

    # calculate gradients for weights and biases
    d_weights = np.dot(new_running_gradient, self.current_input)
    d_biases = new_running_gradient

    # adjust weights and biases
    self.weights = self.weights - d_weights
    self.biases = self.biases - d_biases

    # tack this layer's weights onto running gradient and return
    return np.dot(new_running_gradient, self.weights)

def get_activations_by_name(activation_name):
  if (activation_name == 'tanh'):
    return (utils.tanh, utils.d_tanh)
  if (activation_name == 'relu'):
    return (utils.relu, utils.d_relu)
  if (activation_name == 'softmax'):
    return (utils.softmax, utils.d_softmax)
  return (utils.identity, utils.identity)