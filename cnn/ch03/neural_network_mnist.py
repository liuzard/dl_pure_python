"""利用三层神经网络预测"""
import pickle
import numpy as np
from softmax import softmax
from sigmoid import sigmoid
from cnn.dataset import mnist


class NeuralNetworkMnist(object):
    def __init__(self, weight_path=None):
        self.network_weights = self._init_network_weights(weight_path)

    def _init_network_weights(self, weight_path):
        """初始化网络权重"""
        if weight_path:
            return self._init_weight_file(weight_path)
        else:
            return self._init_weight_random()

    def _init_weight_random(self):
        """随机初始化网络权重"""
        weights_dict = dict()
        weights_dict["w1"] = np.random.random((784, 50))
        weights_dict["b1"] = np.random.random(50)
        weights_dict["w2"] = np.random.random((50, 100))
        weights_dict["b2"] = np.random.random(100)
        weights_dict["w3"] = np.random.random((100, 10))
        weights_dict["b3"] = np.random.random(10)
        return weights_dict

    def _init_weight_file(self, weight_path):
        """通过加载文件初始化网络权重"""
        with open(weight_path, "rb") as fr:
            return pickle.load(fr)

    def forward(self, x):
        """前向传播"""
        a1 = x.dot(self.network_weights["w1"]) + self.network_weights["b1"]
        z1 = sigmoid(a1)
        a2 = z1.dot(self.network_weights["w2"]) + self.network_weights["b2"]
        z2 = sigmoid(a2)
        a3 = z2.dot(self.network_weights["w3"]) + self.network_weights["b3"]
        y = softmax(a3)
        return y


if __name__ == "__main__":
    train_set, test_set = mnist.load_mnist()

    for i in range(100):
        nnm = NeuralNetworkMnist()
        y = nnm.forward(train_set[0])
        y = np.argmax(y, axis=1)
        true_pred = np.sum(y == train_set[1])
        print(np.sum(true_pred) / len(y))
