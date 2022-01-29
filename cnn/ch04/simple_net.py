import numpy as np
from cnn.ch03.softmax import softmax
from cross_entropy import cross_entropy
from gradient_descend import gradient_descent


class SimpleNet(object):
    """简单的神经网络"""

    def __init__(self, weight_init_std=0.01):
        self.W = np.random.randn(784, 10) * weight_init_std

    def forward(self, x):
        a1 = x.dot(self.W)
        return a1

    def loss(self, x, y_true):
        a1 = self.forward(x)
        y_pred = softmax(a1)
        loss = cross_entropy(y_true, y_pred)
        acc = self.cal_acc(x, y_true)
        print(f"loss: {loss}, acc: {acc}")
        return loss

    def cal_acc(self, x, y_true):
        """计算准确率"""
        y_pred = np.argmax(self.forward(x), axis=1)
        return np.sum(y_pred == y_true) / y_pred.shape[0]


if __name__ == "__main__":
    simple_net = SimpleNet()
    x = np.ones([10, 784]) * np.arange(10).reshape(10, 1)
    y_true = np.arange(10)


    def f(W):
        return simple_net.loss(x, y_true)


    gradient_descent(f, simple_net.W)
