import numpy as np


class NeuralNetwork(object):
    """
    A lot of the gist of this code comes from Stephen Welch's youtube videos
    @stephencwelch https://www.youtube.com/watch?v=UJwK6jAStmg
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        defines the parameters of the neural network class. takes information about the
        desired size of the neural network. only supports one hidden layer.
        :param input_size:
        :param hidden_size:
        :param output_size:
        """

        self.input_size = input_size  # number of input nodes
        self.hidden_size = hidden_size  # number of nodes in the hidden layer
        self.output_size = output_size  # number of nodes to return

        # weights connecting the layers are initialized randomly
        self.weight_set_1 = np.random.randn(self.input_size, self.hidden_size)
        self.weight_set_2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, input_data):

        """
        forward pass through the network.
        :param input_data: data for prediction
        :return: a value between 0 and 1 giving the probability of a prediction
        """

        self.layer_2 = np.dot(input_data, self.weight_set_1)  # multiply the input data by our current 'best weights'
        self.activity = self.sigmoid(self.layer_2)  # compute the activation of this weighted layer (val between 0-1)
        self.layer_3 = np.dot(self.activity, self.weight_set_2)  # from the hidden layer, calculate weights to output

        output = self.sigmoid(self.layer_3)  # 'predict' the output of a given input using the activation function

        return output

    def sigmoid(self, layer):
        """
        activation function: decides the amount that a given node contributes to the output
        in this case a sigmoidal activation function is used
        :param layer: array of weighted values
        :return: value between 0 and 1
        """
        return 1/(1+np.exp(-layer))

    def sigmoid_derivative(self, layer):
        """
        the rate of change of the sigmoid activation function
        :param layer: array of values
        :return: derivative of sigmoid function
        """
        return np.exp(-layer)/((1+np.exp(-layer))**2)

    def cost(self, input_data, known_output):
        """
        calculates a simple mean square error cost function
        used by the training functionality to minimize the costs/loss of the network
        :param input_data:
        :param known_output: assigned class labels for the input data
        :return: 1/2 the cost
        """
        self.output = self.forward(input_data)  # predict the output
        cost = 0

        # for every value in our output arrays, calculate the difference between the predicted and true value
        # to do a simple batch gradient, sum all the errors
        for i in range(known_output.shape[0]):
            for j in range(known_output.shape[1]):
                cost += (known_output[i,j] - self.output[i,j]) ** 2

        return cost*0.5  # return 1/2 total cost to make computation easier

    def cost_function_derivative(self, input_data, known_output):
        """
        to get the gradient, or an indication of which way to minimize the errors, compute the rate of change of
        the cost function.
        :param input_data:
        :param known_output:
        :return: the rate of change of layer 2 and layer 3
        """
        self.output = self.forward(input_data)  # predict the output

        delta3 = np.multiply(-(known_output - self.output), self.sigmoid_derivative(self.layer_3))  # change of layer 3
        delta2 = np.dot(delta3, self.weight_set_2.T) * self.sigmoid_derivative(self.layer_2)  # change of layer 2
        return np.dot(input_data.T, delta2), np.dot(self.activity.T, delta3)  # the derivative of the cost function w/ respect to the change

    def set_weights(self, weights):
        """
        set weights given values from the training gradient optimization
        :param weights: new weights for the network
        :return: none
        """
        self.weight_set_1 = np.reshape(weights[0:self.hidden_size * self.input_size], (self.input_size, self.hidden_size))
        self.weight_set_2 = np.reshape(weights[self.hidden_size * self.input_size:self.hidden_size * self.input_size + self.hidden_size * self.output_size],
                                       (self.hidden_size, self.output_size))

    def get_weights(self):
        """
        get weights of the system and format them nicely for the gradient optimizer
        :return:
        """
        return np.concatenate((self.weight_set_1.ravel(), self.weight_set_2.ravel()))  # concatenate, make 1D

    def get_gradient(self, input_data, known_output):
        """
        returns gradient of system- the cost function derivative for the current hyperparameters
        :param input_data:
        :param known_output:
        :return: the current gradients
        """
        der1, der2 = self.cost_function_derivative(input_data, known_output)
        return np.concatenate((der1.ravel(), der2.ravel()))  # concatenate and make 1D
