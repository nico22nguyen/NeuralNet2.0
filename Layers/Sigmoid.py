import numpy as np

class Sigmoid:
  def __init__(self):
    self.sig_activation = None

  def forward_pass(self, _input: list):
    self.sig_activation = 1/(1 + np.exp(-_input))
    return self.sig_activation

  # returns new "running gradient"
  def back_propagate(self, running_gradient):
    return self.sig_activation * (1 - self.sig_activation) * running_gradient

  def update(self):
    pass