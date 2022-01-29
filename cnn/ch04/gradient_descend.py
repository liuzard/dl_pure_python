"""梯度下降算法"""
import numpy as np
from numerical_diff import numerical_gradient


def gradient_descent(f, init_x, lr=0.1, step=100):
    """梯度下降算法"""
    x = init_x
    for i in range(step):
        gradients = numerical_gradient(f, x)
        x -= lr * gradients
    return x


if __name__ == "__main__":
    def function_1(x):
        return x[0] ** 2 + x[1] ** 2


    print(gradient_descent(function_1, np.array([2.0, 3.0])))
