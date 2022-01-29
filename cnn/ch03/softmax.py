"""softmax函数"""
import numpy as np


def softmax_native(x):
    """朴素的softmax，在x的元素比较大的时候容易溢出"""
    return np.exp(x) / np.sum(np.exp(x))


# def softmax(x):
#     x_max = np.max(x)
#     x = x - x_max
#     return np.exp(x) / np.sum(np.exp(x))


# def softmax(x):
#     if x.ndim == 2:
#         # x = x.T
#         x = x - np.max(x, axis=1).reshape(-1, 1)
#         y = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
#         return y
#
#     x = x - np.max(x)  # 溢出对策
#     return np.exp(x) / np.sum(np.exp(x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(softmax_native(x))
    print(softmax(x))

    x = [1000, 2000, 3000, 4000]
    print(softmax_native(x))
    print(softmax(x))
