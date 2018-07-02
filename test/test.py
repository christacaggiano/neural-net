from main import neural_net
from data.utils import one_hot_encode
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from neural_network.encoder import NeuralNetwork
from mock import patch


class TestNeuralNet(unittest.TestCase):

    def test_neural_net_empty(self):
        x_train = np.zeros((1,1), dtype=float)
        y_train = np.zeros((1,1), dtype=float)
        x = np.zeros((1,1), dtype=float)

        self.assertEqual(neural_net(x_train, y_train, x, hidden_size=1).shape, (1, 1))

    def test_neural_net_ones(self):
        x_train = np.full((1, 1), 1, dtype=float)
        y_train = np.full((1, 1), 1, dtype=float)
        x = np.full((1, 1), 1, dtype=float)

        self.assertEqual(neural_net(x_train, y_train, x, hidden_size=1).shape, (1, 1))

    def test_encoding(self):

        encoded_strings = [one_hot_encode(s) for s in ["A", "T", "C", "G"]]
        assert_array_equal(encoded_strings, [[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]]])

    def test_sigmoid(self):
        layer = np.zeros((1, 10))
        assert_array_equal(NeuralNetwork.sigmoid(layer), [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

    def test_sigmoid_ones(self):
        layer = np.full((1, 10), 1)
        assert_array_almost_equal(NeuralNetwork.sigmoid(layer), [[0.73105858,  0.73105858,  0.73105858,  0.73105858,
                                                            0.73105858,  0.73105858,0.73105858,  0.73105858,
                                                            0.73105858,  0.73105858]])

    def test_sigmoid_derivative(self):
        layer= np.full((1, 10), 0)
        assert_array_equal(NeuralNetwork.sigmoid_derivative(layer), [[ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25]])

    @patch.object(NeuralNetwork, 'forward', lambda x,y: np.identity(2))
    def test_cost_fn_equal(self):
        test_obj = NeuralNetwork(1,3,1)
        self.assertEqual(test_obj.cost(None, np.identity(2)), 0)

    @patch.object(NeuralNetwork, 'forward', lambda x,y: np.array([[1,2], [3,4]]))
    def test_cost_fn_sequential(self):
        test_obj = NeuralNetwork(1,3,1)
        self.assertEqual(test_obj.cost(None, np.array([[0,0], [0,0]])), 15)



if __name__ == '__main__':
    unittest.main()