import numpy as np
from neural_net import neuralNetwork
from train import Train


def round_matrix(matrix, decimal_number):
    return np.round(matrix, decimals=decimal_number)


def encoder(matrix_size):

    x = np.identity(matrix_size)
    y = np.identity(matrix_size)

    NN = neuralNetwork(input_size=matrix_size, output_size=matrix_size, hidden_size=3, number_layers=1)

    T = Train(NN)
    T.train(x, y)

    predict = NN.forward(x)

    return round_matrix(predict, 1)

if __name__ == "__main__":
    print(encoder(3))