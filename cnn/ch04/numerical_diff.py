"""数值微分"""
import numpy as np


def numerical_diff_native(f, x):
    """根据数值微分定义计算导数"""
    h = 1e-4
    return (f(x + h) - f(x)) / h


def numerical_diff(f, x):
    """根据中心差分计算导数"""
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    """数值梯度"""
    h = 1e-4
    gradients = np.zeros_like(x)
    for i in range(x.shape[0]):
        tmp_val = x[i]
        x[i] = tmp_val + h

        f_h1 = f(x)
        x[i] = tmp_val - h

        f_h2 = f(x)
        gradients[i] = (f_h1 - f_h2) / (2 * h)
        x[i] = tmp_val
    return gradients


if __name__ == "__main__":
    def function_1(x):
        return x ** 2 + 2 * x


    def function_2(x):
        return x[0] ** 2 + x[1] ** 2


    print(numerical_diff_native(function_1, 5))
    print(numerical_diff(function_1, 5))
    gradients = numerical_gradient(function_2, np.array([0.0, 4.0]))
    print(gradients)
