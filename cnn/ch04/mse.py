"""均方误差损失函数"""
import numpy as np


def mse(y_true, y_pred):
    """均方差损失函数"""
    return 0.5 * np.sum(np.square(y_true - y_pred))


if __name__ == "__main__":
    y_true = np.array([[0, 1, 0], [1, 0, 0]])
    y_pred = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])
    print(mse(y_true, y_pred))
