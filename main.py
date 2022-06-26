from keras.datasets import mnist
import numpy as np

from Model import Model
from Layer import Layer
from utils import one_hot

# load data
(train_X_raw, train_Y_raw), (test_X_raw, test_Y_raw) = mnist.load_data()

# determine input size of model
INPUT_SIZE = train_X_raw.shape[1] * train_X_raw.shape[2]

# reshape train data
train_X = np.reshape(train_X_raw, (train_X_raw.shape[0], INPUT_SIZE))
train_X = train_X / 255
train_Y = [one_hot(vect, 10) for vect in train_Y_raw]

# reshape test data
test_X = np.reshape(test_X_raw, (test_X_raw.shape[0], INPUT_SIZE))
test_X = test_X / 255
test_Y = [one_hot(vect, 10) for vect in test_Y_raw]

# create model
model = Model()
model.add_layer(Layer(100, INPUT_SIZE, 'relu'))
model.add_layer(Layer(10, 100, 'sigmoid'))

model.train(train_X, train_Y, test_X, test_Y)
model.test(test_X, test_Y)