import numpy as np
import scipy.optimize


class neuralNetwork(object):
    """
    A lot of the gist of this code comes from Stephen Welch's youtube videos
    @stephencwelch https://www.youtube.com/watch?v=UJwK6jAStmg
    """
    def __init__(self, input_size, hidden_size, output_size, number_layers):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = number_layers

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

    # def cost(self, input_data, known_output):
    #     self.output = self.forward(input_data)
    #     cost = 0
    #     for i in range(known_output.shape[0]):
    #         for j in range(known_output.shape[1]):
    #             cost += (known_output[i,j] - self.output[i,j]) ** 2
    #     return cost*0.5

    def loss(self, input_data, known_output):
        self.output = self.forward(input_data)
        self.output = np.clip(self.output, 1e-15, 1 - 1e-15)  # Avoid division by zero
        return - known_output * np.log(self.output) - (1 - known_output) * np.log(1 - self.output)

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

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self.forward(X)
        loss = np.mean(self.loss(y, y_pred))
        acc = self.acc(y, y_pred)

        return loss, acc

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def accuracy_score(self, y_true, y_pred):
        """ Compare y_true to y_pred and return the accuracy """
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy

    def acc(self, y, p):
        return self.accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

