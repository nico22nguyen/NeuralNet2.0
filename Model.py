import numpy as np

from utils import mean_squared_error, d_mean_squared_error

class Model:
  def __init__(self):
    self.layers = []

  def add_layer(self, layer):
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
    for layer in self.layers:
      layer.update()

  def train(self, input_matrix, expected_results, validation_inputs, validation_expected):
    num_correct = 0
    for i, (_input, expected_result) in enumerate(zip(input_matrix, expected_results)):
      # add extra dimension to input to make matrix math work out
      input = np.expand_dims(_input, 0)

      # compute result of input
      actual_result = self.evaluate(input)
      print(np.argmax(actual_result))

      if np.argmax(expected_result) == np.argmax(actual_result):
        num_correct += 1

      # compute loss of result vs expected
      loss = mean_squared_error(expected_result, actual_result)
      d_loss = d_mean_squared_error(expected_result, actual_result)
      if i % 10000 == 0:
        print('Loss on sample ', i, ': ', np.linalg.norm(loss))
      # perform gradient descent
      self.gradient_descent(d_loss)
    
    print('num_correct: ', num_correct)
    print('Accuracy: ', round(num_correct / input_matrix.shape[0], 4) * 100, '%')
    
  
  def test(self, input_matrix, expected_results):
    num_correct = 0
    for _input, expected_result in zip(input_matrix, expected_results):
      # add extra dimension to input to make matrix math work out
      input = np.expand_dims(_input, 0)
      # compute result of input
      actual_result = self.evaluate(input)

      if np.argmax(expected_result) == np.argmax(actual_result):
        num_correct += 1
    
    print('Accuracy: ', round(num_correct / input_matrix.shape[0], 4) * 100, '%')