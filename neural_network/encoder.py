import numpy as np


class NeuralNetwork(object):
    """
    A lot of the gist of this code comes from Stephen Welch's youtube videos
    @stephencwelch https://www.youtube.com/watch?v=UJwK6jAStmg
    """
    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weight_set_1 = np.random.randn(self.input_size, self.hidden_size)
        self.weight_set_2 = np.random.randn(self.hidden_size, self.output_size)



    def forward(self, input_data):

        self.layer_2 = np.dot(input_data, self.weight_set_1)
        self.activity = self.sigmoid(self.layer_2)
        self.layer_3 = np.dot(self.activity, self.weight_set_2)

        output = self.sigmoid(self.layer_3)

        return output

    def sigmoid(self, layer):
        return 1/(1+np.exp(-layer))

    def sigmoid_derivative(self, layer):
        return np.exp(-layer)/((1+np.exp(-layer))**2)

    def cost(self, input_data, known_output):
        self.output = self.forward(input_data)
        cost = 0
        for i in range(known_output.shape[0]):
            for j in range(known_output.shape[1]):
                cost += (known_output[i,j] - self.output[i,j]) ** 2
        return cost*0.5

    def cost_function_derivative(self, input_data, known_output):
        self.output = self.forward(input_data)
        delta3 = np.multiply(-(known_output - self.output), self.sigmoid_derivative(self.layer_3))
        dJdW2 = np.dot(self.activity.T, delta3)
        delta2 = np.dot(delta3, self.weight_set_2.T) * self.sigmoid_derivative(self.layer_2)
        dJdW1 = np.dot(input_data.T, delta2)
        return dJdW1, dJdW2

    def set_weights(self, weights):
        W1_start = 0
        W1_end = self.hidden_size * self.input_size
        self.weight_set_1 = np.reshape(weights[W1_start:W1_end], (self.input_size, self.hidden_size))
        W2_end = W1_end + self.hidden_size * self.output_size
        self.weight_set_2 = np.reshape(weights[W1_end:W2_end], (self.hidden_size, self.output_size))

    def get_weights(self):
        return np.concatenate((self.weight_set_1.ravel(), self.weight_set_2.ravel()))

    def get_gradient(self, input_data, known_output):
        dJdW1, dJdW2 = self.cost_function_derivative(input_data, known_output)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
