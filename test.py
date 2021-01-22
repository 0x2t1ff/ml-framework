#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from layers import *
from activations import * 
from losses import *
from optimizers import *
from models import Model
from accuracy import *

nnfs.init()

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()
model.add(Dense_Layer(2, 512, weight_regularizer_l2=5e-4,
                             bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Dropout_Layer(0.1))
model.add(Dense_Layer(512, 3))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()

)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

