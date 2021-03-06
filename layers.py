import numpy as np

class Dense_Layer:
  
  def __init__(self, n_inputs, n_neurons, 
               weight_regularizer_l1=0, weight_regularizer_l2=0,
               bias_regularizer_l1=0, bias_regularizer_l2=0):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    self.weight_regularizer_l1 = weight_regularizer_l1
    self.weight_regularizer_l2 = weight_regularizer_l2
    self.bias_regularizer_l1 = bias_regularizer_l1
    self.bias_regularizer_l2 = bias_regularizer_l2

    
  def forward(self, inputs, training):
    self.output = np.dot(inputs, self.weights) + self.biases
    self.inputs = inputs

  def backward(self, dvalues):
    # Gradients on parameters
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

    # Gradients on regularization
    # L1 on weights
    if self.weight_regularizer_l1 > 0:
      dL1 = np.ones_like(self.weights)
      dL1[self.weights < 0] = -1
      self.dweights += self.weight_regularizer_l1 * dL1

    # L2 on weights
    if self.weight_regularizer_l2 > 0:
      self.dweights += 2 * self.weight_regularizer_l2 * \
                        self.weights
     
    # L1 on biases
    if self.bias_regularizer_l1 > 0:
      dL1 = np.ones_like(self.biases)
      dL1[self.biases < 0] = -1
      self.dbiases += self.bias_regularizer_l1 * dL1

    # L2 on biases
    if self.bias_regularizer_l2 > 0:
      self.dbiases += 2 * self.bias_regularizer_l2 * \
                        self.biases
    

    self.dinputs = np.dot(dvalues, self.weights.T)



class Dropout_Layer:
  
  def __init__(self, rate):
    # Store rate, we invert it as for example for dropout
    # of 0.1 we need success rate of 0.9
    self.rate = 1 - rate

  def forward(self, inputs, training):
    # Save input values
    self.inputs = inputs

    if not training:
      self.output = inputs.copy()
      return
    
    # Generate and save scaled mask
    self.binary_mask = np.random.binomial(1, self.rate,
                       size=inputs.shape) / self.rate

    self.output = inputs * self.binary_mask

  def backward(self, dvalues):
    # Gradient on values
    self.dinputs = dvalues * self.binary_mask
                       

class Input_Layer:
  def forward(self, inputs, training):
    self.output = inputs
