from utils import mean_squared_error
from Layer import Layer

class Model:
  def __init__(self):
    self.layers: list[Layer] = []

  def add_layer(self, layer: Layer):
    self.layers.append(layer)

  def evaluate(self, inputs):
    # initialize "result" to inputs
    result = inputs
    # loop through layers and feed results forward
    for layer in self.layers:
      result = layer.forward_pass(result)
    
    # return result of network
    return result

  def gradient_descent(self, loss):
    running_gradient = loss
    for layer in reversed(self.layers):
      running_gradient = layer.back_propagate(running_gradient)

  def train(self, input_matrix, expected_results):
    for input, expected_result in zip(input_matrix, expected_results):
      # compute result of input
      actual_result = self.evaluate(input)

      # compute loss of result vs expected
      loss = mean_squared_error(actual_result, expected_result)

      # perform gradient descent
      self.gradient_descent(loss)
  
  def test(self, input_matrix, expected_results):
    for input, expected_result in zip(input_matrix, expected_results):
      # compute result of input
      actual_result = self.evaluate(input)

      # show actual vs expected
      print('expected_result: ', expected_result)
      print('actual_result: ', actual_result)

  





