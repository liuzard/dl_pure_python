"""三层神经网络"""

import numpy as np
from sigmoid import sigmoid
from identity import identity_function


class ThreeLayerNetwork:
    def __init__(self):
        self.network = self._init_network_weights()

    def _init_network_weights(self):
        """初始化网络权重"""
        network = dict()
        network['w1'] = np.array([[1, 2, 3], [4, 5, 6]])
        network['b1'] = np.array([1, 2, 3])
        network['w2'] = np.array([[1, 2], [3, 4], [5, 6]])
        network['b2'] = np.array([1, 2])
        network['w3'] = np.array([[1, 2], [3, 4]])
        network['b3'] = np.array([1, 2])
        return network

    def forword(self, x):
        """前向传播"""
        a1 = x.dot(self.network["w1"]) + self.network['b1']
        z1 = sigmoid(a1)
        a2 = z1.dot(self.network['w2']) + self.network['b2']
        z2 = sigmoid(a2)
        a3 = z2.dot(self.network['w3']) + self.network['b3']
        z3 = identity_function(a3)
        return z3


if __name__ == "__main__":
    tln = ThreeLayerNetwork()
    y = tln.forword(np.array([1, 2]))
    print(y)
