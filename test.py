#!/usr/bin/env python3
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from layers import *
from activations import * 
from losses import *
from optimizers import *

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)

dense1 = Dense_Layer(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Dropout_Layer(0.1)
dense2 = Dense_Layer(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(decay=5e-5, learning_rate=0.05)

for epoch in range(10001):
  dense1.forward(X)
  activation1.forward(dense1.output)
  dropout1.forward(activation1.output)
  dense2.forward(dropout1.output)
  # Calculate loss from output of activation2 so softmax activation
  data_loss = loss_activation.forward(dense2.output, y)

  # Calculate regualrization penalty
  regularization_loss = loss_activation.regularization_loss(dense1) + \
                        loss_activation.regularization_loss(dense2)
  loss = data_loss + regularization_loss

  predictions = np.argmax(loss_activation.output, axis=1)
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  accuracy = np.mean(predictions == y)
  
  if not epoch % 100:
    print(f'epoch: {epoch}, ' +
          f'acc: {accuracy:.3f}, ' + 
          f'loss: {loss:.3f}, ' +
          f'learning rate: {optimizer.current_learning_rate:.3f},' )
  
  loss_activation.backward(loss_activation.output, y)
  dense2.backward(loss_activation.dinputs)
  dropout1.backward(dense2.dinputs)
  activation1.backward(dropout1.dinputs)
  dense1.backward(activation1.dinputs)

  optimizer.pre_update_params()
  optimizer.update_params(dense1)
  optimizer.update_params(dense2)
  optimizer.post_update_params()

X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)

accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')