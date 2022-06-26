class Relu:
  def __init__(self):
    self.relu_activation = None

  def forward_pass(self, _input: list):
    self.relu_activation = (_input > 0) * 1
    return self.relu_activation * _input

  # returns new "running gradient"
  def back_propagate(self, running_gradient):
    return self.relu_activation * running_gradient

  def update(self):
    pass