import numpy as np
from activations import *

class Loss:

  def remember_trainable_layers(self, trainable_layers):
    self.trainable_layers = trainable_layers


  def calculate(self, output, y, *, include_regularization=False):
    sample_losses = self.forward(output, y)

    data_loss = np.mean(sample_losses)

    if not include_regularization:
      return data_loss

    return data_loss, self.regularization_loss()

  def regularization_loss(self):
   
    # 0 by default
    regularization_loss = 0

    for layer in self.trainable_layers:
      # L1 regularization - weights
      # calculate only when factor greater than 0
      if layer.weight_regularizer_l1 > 0:
        regularization_loss += layer.weight_regularizer_l1 * \
                               np.sum(np.abs(layer.weights))
      
      # L2 regularization - weights 
      if layer.weight_regularizer_l2 > 0:
        regularization_loss += layer.weight_regularizer_l2 * \
                               np.sum(layer.weights * layer.weights)
      
      # L1 regularization - biases
      # calculate only when factor greater than 0
      if layer.bias_regularizer_l1 > 0:
        regularization_loss += layer.bias_regularizer_l1 * \
                               np.sum(np.abs(layer.biases))
      
      # L2 regularization - biases 
      if layer.bias_regularizer_l2 > 0:
        regularization_loss += layer.bias_regularizer_l2 * \
                               np.sum(layer.biases * layer.biases)

    return regularization_loss
    
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped*y_true,
            axis=1
            )
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(lables)[y_true]
            
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        

class Activation_Softmax_Loss_CategoricalCrossEntropy(Loss):
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):
  
  def forward(self, y_pred, y_true):
    
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Calculate sample-wise loss
    sample_losses = -(y_true * np.log(y_pred_clipped) +
                    (1 - y_true) * np.log(1 - y_pred_clipped))
  
    sample_losses = np.mean(sample_losses, axis=-1)

    return sample_losses


  def backward(self, dvalues, y_true):
    
    # Number of samples
    samples = len(dvalues)
    # Number of outputs in every sample
    # We'll use the first sample ot count them
    outputs = len(dvalues[0])

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

    # Calculate grandient
    self.dinputs = -(y_true / clipped_dvalues - 
                     (1 - y_true) / (1 - clipped_dvalues)) / outputs 

    self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):
  
  def forward(self, y_pred, y_true):
    # Calculcate loss
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
    return sample_losses

  def backward(self, dvalues, y_true):
    
    # Number of samples
    samples = len(dvalues)
    # Number of ouputs in every sample
    # We'll use the first sample to count them
    outputs = len(dvalues[0])
    
    # Gradient on values
    self.dinputs = -2 * (y_true - dvalues) / outputs
    # Normalize gradient
    self.dinputs = self.dinputs / samples



class Loss_MeanAbsoluteError(Loss):
  
  def forward(self, y_pred, y_true):
    # Calculcate loss
    sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
    return sample_losses

  def backward(self, dvalues, y_true):
    
    # Number of samples
    samples = len(dvalues)
    # Number of ouputs in every sample
    # We'll use the first sample to count them
    outputs = len(dvalues[0])
    
    # Gradient on values
    self.dinputs = np.sign(y_true - dvalues) / outputs
    # Normalize gradient
    self.dinputs = self.dinputs / samples
