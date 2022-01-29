"""双层申请网络"""
import numpy as np
from cross_entropy import cross_entropy_oh, cross_entropy_error
from cnn.ch03.softmax import softmax
from cnn.ch03.sigmoid import sigmoid
from numerical_diff import numerical_gradient
from cnn.dataset import mnist


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, init_weight_std=0.01):
        """初始化参数"""
        self.params = dict()
        self.params["W1"] = init_weight_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = init_weight_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def foward(self, x):
        a1 = x.dot(self.params["W1"]) + self.params["b1"]
        z1 = sigmoid(a1)
        a2 = z1.dot(self.params["W2"]) + self.params["b2"]
        y = softmax(a2)
        return y

    def loss(self, x, y_true):
        y_pred = self.foward(x)
        return cross_entropy_error(y_true, y_pred)

    def acc(self, x, y_true):
        y_pred = self.foward(x)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.sum(y_pred == y_true) / y_true.shape[0]

    def numerical_gradient(self, x, y_true):
        f = lambda W: self.loss(x, y_true)
        grads = dict()
        grads['W1'] = numerical_gradient(f, self.params["W1"])
        grads['b1'] = numerical_gradient(f, self.params["b1"])
        grads['W2'] = numerical_gradient(f, self.params["W2"])
        grads['b2'] = numerical_gradient(f, self.params["b2"])
        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


if __name__ == "__main__":
    two_layer_net = TwoLayerNet(784, 50, 10)

    train_set, test_set = mnist.load_mnist(one_hot_label=True)
    train_imgs, train_lables = train_set
    test_imgs, test_labels = test_set
    train_size = train_imgs.shape[0]
    batch_size = 100
    lr = 0.1
    for i in range(10000):
        random_index = np.random.choice(train_size, batch_size)
        train_x, train_y = train_imgs[random_index], train_lables[random_index]

        for key in ["W1", "W2", "b1", "b2"]:
            two_layer_net.params[key] -= lr * two_layer_net.numerical_gradient(train_x, train_y)[key]
        if i % 10 == 0:
            loss_test = two_layer_net.loss(test_imgs, test_labels)
            acc_test = two_layer_net.acc(test_imgs, test_labels)
            print(f"step {i},test loss {loss_test},test acc {acc_test}")
