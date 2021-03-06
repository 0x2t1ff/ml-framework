import numpy as np

class Activation_ReLU:

  def forward(self, inputs, training):
    self.inputs = inputs
    self.output = np.maximum(inputs, 0)

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0


  def predictions(self, outputs):
    return outputs

class Activation_Softmax:

  def forward(self, inputs, training):
    self.inputs = inputs
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      single_output = single_output.reshape(-1, 1)
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

  def predictions(self, outputs):
    return np.argmax(outputs, axis=1)

class Activation_Sigmoid:
  
  def forward(self, inputs, training):
    # Save input and calculate/save output
    # of the sigmoid function
    self.inputs = inputs
    self.output = 1 / (1 + np.exp(-inputs))

  def backward(self, dvalues):
    # Derivative - calculates from output of the sigmoid function
    self.dinputs = dvalues * (1 - self.output) * self.output

  def predictions(self, outputs):
    return (outputs > 0.5) * 1


class Activation_Linear:

  def forward(self, inputs, training):
    # Just remember values
    self.inputs = inputs
    self.output = inputs

  def backward(self, dvalues):
    # derivaitve is 1, 1 * dvalues = dvalues - the chain rule
    self.dinputs = dvalues.copy()


  def predictions(self, outputs):
    return outputs
