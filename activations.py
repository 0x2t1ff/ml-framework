import numpy as np

class Activation_ReLU:

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(inputs, 0)

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      single_output = single_output.reshape(-1, 1)
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:
  
  def forward(self, inputs):
    # Save input and calculate/save output
    # of the sigmoid function
    self.inputs = inputs
    self.output = 1 / (1 + np.expj(-inputs))

  def backward(self, dvalues):
    # Derivative - calculates from output of the sigmoid function
    self.dinputs = dvalues * (1 - self.output) * self.output
