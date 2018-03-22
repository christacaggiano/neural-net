import numpy as np
import scipy.optimize
from neural_net import neuralNetwork


class Train(object):
    def __init__(self, neural_net):
        self.N = neural_net

    def training_function(self, weights, input_data, known_output):
        self.N.set_weights(weights)
        cost = self.N.cost(input_data, known_output)
        gradient = self.N.get_gradient(input_data, known_output)
        return cost, gradient

    def train(self, input_data, known_output):
        self.input_data = input_data
        self.known_output = known_output
        initial_weights = self.N.get_weights()
        options = {'maxiter': 7000, 'disp': False}

        self.training_function(initial_weights, input_data, known_output)

        optim = scipy.optimize.minimize(self.training_function, initial_weights, jac=True, method="Newton-CG",
                                        args=(input_data, known_output), options=options)
        self.N.set_weights(optim.x)
        self.optimization = optim
