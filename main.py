import numpy as np
from neural_network.encoder import NeuralNetwork
from neural_network.train import Train
from data.utils import return_data, print_predictions


def round_matrix(matrix, decimal_number):
    """
    rounds output matrix to a given decimal place
    :param matrix:
    :param decimal_number: int
    :return:
    """
    return np.round(matrix, decimals=decimal_number)

def autoencoder(matrix_size):
    """
    takes a given matrix size, creates its identity matrix and learns a NN based on it
    :param matrix_size: int
    :return:
    """

    x = np.identity(matrix_size)
    y = np.identity(matrix_size)

    NN = NeuralNetwork(input_size=x.shape[1], output_size=y.shape[1], hidden_size=3)  # initializes a 3 node hidden layer NN
    T = Train(NN)  # sets up training object
    T.train(x, y)  # trains given the identity matrices

    predict = NN.forward(x)  # predicts based on just the x data, output should be identical

    return round_matrix(predict, 1)  # return rounded prediction


def neural_net(x_train, y_train, x_test, hidden_size):
    """
    calls neural network, for use on the transcription factor binding data
    :param x_train: input training
    :param y_train: output training
    :param x_test: data for testing
    :return: rounded prediction matrix
    """

    NN = NeuralNetwork(input_size=x_train.shape[1], output_size=y_train.shape[1], hidden_size=hidden_size )  # initialize NN

    T = Train(NN)  # training object
    T.train(x_train, y_train)  # train model with our set aside training data

    predict = NN.forward(x_test)  # predict output for our test data

    return round_matrix(predict, 3)  # return the prediciton matrix


def cross_validate(percent_validation, x, y):
    """
    pick a subset of data for training, leave rest for testing
    :param percent_validation: amount of data to be set aside for training
    :param x: input data
    :param y: output data
    :return: x and y training, x and y testing
    """
    assert(percent_validation <= 1)
    mask = np.random.choice([True, False], size=x.shape[0], p=[1-percent_validation, percent_validation])  # picks data to be subset s
    x_training = x[mask]  # all the data that is to be subset
    y_training = y[mask]
    x_validation = x[np.invert(mask)]  # remaining data for testing
    y_validation = y[np.invert(mask)]

    return x_training, y_training, x_validation, y_validation


def run():
    """
    runs neural network
    :return: binding predictions
    """

    hidden_size = 3

    print(autoencoder(8))  # test autoencoder

    x, y, x_test = return_data(num_subsample=500)  # subsample negative data, get real sequence information

    x_training, y_training, x_validation, y_validation = cross_validate(0.2, x, y)  # pick testing and training data

    print(neural_net(x_training, y_training, x_validation, hidden_size))  # print results of NN

    return neural_net(x_training, y_training, x_test, hidden_size)

if __name__ == "__main__":
    predictions = run()
    print_predictions(predictions)
