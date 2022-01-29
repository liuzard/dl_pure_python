"""交叉熵损失函数"""
import numpy as np


def cross_entropy_native(y_true, y_pred):
    """交叉熵损失函数，单条样本"""
    delta = 1e-7
    return -np.sum(y_true * np.log(y_pred + delta))


def cross_entropy_oh(y_true, y_pred):
    """交叉熵，批量计算，true标签为one-hot形式"""
    delta = 1e-7
    if y_pred.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)
    return -np.sum(y_true * np.log(y_pred + delta)) / y_pred.shape[0]


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def cross_entropy(y_true, y_pred):
    """交叉熵，批量计算，true标签为数值id形式下"""
    delta = 1e-7
    if y_pred.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)
    # print(y_pred[:, y_true])
    return -np.sum(np.log(y_pred[np.arange(y_true.size), y_true] + delta)) / y_true.shape[0]


if __name__ == "__main__":
    # 单个样本的交叉熵计算
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0])
    loss = cross_entropy_native(y_true, y_pred)
    print(loss)

    # 多个样本的交叉熵计算
    y_true_batch = np.array([[0, 0, 1], [0, 0, 1]])
    y_pred_batch = np.array([[0.9, 0.1, 0], [0.9, 0.1, 0]])
    loss_batch = cross_entropy_oh(y_true_batch, y_pred_batch)
    print(loss_batch)

    # 多个样本交叉熵计算
    y_true_batch = np.array([2, 2])
    y_pred_batch = np.array([[0.9, 0.1, 0], [0.9, 0.1, 0]])
    loss_batch = cross_entropy(y_true_batch, y_pred_batch)
    print(loss_batch)
