import numpy as np

from neural_network.encoder import NeuralNetwork
from neural_network.train import Train
from data.one_hot_encode import return_data


def round_matrix(matrix, decimal_number):
    return np.round(matrix, decimals=decimal_number)

def autoencoder(matrix_size):

    x = np.identity(matrix_size)
    y = np.identity(matrix_size)

    NN = NeuralNetwork(input_size=x.shape[1], output_size=y.shape[1], hidden_size=3)
    T = Train(NN)
    T.train(x, y)

    predict = NN.forward(x)

    return round_matrix(predict, 1)


def neural_net(x_train, y_train, x_test, y_test):



    NN = NeuralNetwork(input_size=x.shape[1], output_size=y.shape[1], hidden_size=3)

    T = Train(NN)
    T.train(x_training, y_train)

    predict = NN.forward(x_test)

    return round_matrix(predict, 3)

def cross_validate(percent_validation, x, y):
    assert(percent_validation <= 1)
    mask = np.random.choice([True, False], size=x.shape[0], p=[1-percent_validation, percent_validation])
    x_training = x[mask]
    y_training = y[mask]
    x_validation = x[np.invert(mask)]
    y_validation = y[np.invert(mask)]

    return x_training, y_training, x_validation, y_validation
if __name__ == "__main__":
    print(autoencoder(8))

    x, y = return_data(num_subsample=100)

    x_training, y_training, x_validation, y_validation = cross_validate(0.2, x, y)
    print(neural_net(x_training, y_training, x_validation, y_validation))

