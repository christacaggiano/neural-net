import scipy.optimize


class Train(object):
    def __init__(self, neural_net):
        """
        initialize training object
        :param neural_net: needs the neural network object and functions
        """
        self.N = neural_net

    def training_function(self, weights, input_data, known_output):
        """
        function to be optimized by scipy optimize
        :param weights: current weights
        :param input_data:
        :param known_output: training data
        :return: the cost and gradient from NN
        """
        self.N.set_weights(weights)  # using NN function, set weights
        cost = self.N.cost(input_data, known_output)  # calculate the cost
        gradient = self.N.get_gradient(input_data, known_output)  # calculate the gradient
        return cost, gradient

    def train(self, input_data, known_output):
        """
        finally, train the network! uses scipy optimization to perform batch gradient descent
        :param input_data:
        :param known_output:
        :return:
        """

        # intialize training data
        self.input_data = input_data
        self.known_output = known_output

        # set initial weights, determined randomly on first iteration
        initial_weights = self.N.get_weights()

        # set max number of iterations for the optimizer
        options = {'maxiter': 7000, 'xtol':1e-8,'disp': True}

        # set the training function
        self.training_function(initial_weights, input_data, known_output)

        # call the scipy optimizer with Newton nonlinear conjugate gradient method, using the jacobian
        optim = scipy.optimize.minimize(self.training_function, initial_weights, jac=True, method="Newton-CG",
                                        args=(input_data, known_output), options=options)

        self.N.set_weights(optim.x)  # set the weights as the trained optimizer output
        self.optimization = optim  # yay heres our optimized results cool
