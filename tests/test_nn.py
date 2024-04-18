import pytest
import numpy as np
from neuralnet import NeuralNetwork, read_data



X, y = read_data('housing.csv')
X_train, y_train = X[:405], y[:405] # 405 Values
X_test, y_test = X[405:], y[405:] # 101 Values


def test_nn_1():
    NN = NeuralNetwork(13, 8, 1)

    NN.train(X_train, y_train, 5000, 0.001)

    MSE = NN.evaluate(X_test, y_test) # MSE for mean squared error
    RSE = np.sqrt(MSE) # RSE for root squared error
    assert round(RSE, 2) == 5.11


def test_nn_2():
    NN = NeuralNetwork(13, 8, 1)

    NN.train(X_train, y_train, 500, 0.001)

    MSE = NN.evaluate(X_test, y_test) # MSE for mean squared error
    RSE = np.sqrt(MSE) # RSE for root squared error
    assert round(RSE, 2) == 3.69


def test_nn_3():
    NN = NeuralNetwork(13, 8, 1)

    NN.train(X_train, y_train, 50, 0.001)

    MSE = NN.evaluate(X_test, y_test) # MSE for mean squared error
    RSE = np.sqrt(MSE) # RSE for root squared error
    assert round(RSE, 2) == 8.3

