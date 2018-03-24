from main import neural_net
import unittest
import numpy as np


class TestNeuralNet(unittest.TestCase):

    def test_neural_net_empty(self):
        x_train = np.zeros((1,1), dtype=float)
        y_train = np.zeros((1,1), dtype=float)
        x = np.zeros((1,1), dtype=float)

        assert neural_net(x_train, y_train, x, hidden_size=1).shape == (1,1)

    def test_neural_net_ones(self):
        x_train = np.full((1, 1), 1, dtype=float)
        y_train = np.full((1, 1), 1, dtype=float)
        x = np.full((1, 1), 1, dtype=float)

        assert neural_net(x_train, y_train, x, hidden_size=1).shape == (1,1)


if __name__ == '__main__':
    unittest.main()