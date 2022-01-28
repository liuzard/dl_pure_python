"""交叉熵损失函数"""
import numpy as np


def cross_entropy(y_true, y_pred):
    delta = 1e-7
    return -np.sum(y_true * np.log(y_pred + delta))


def cross_entropy_batch(y_true, y_pred):
    """交叉熵，批量计算"""
    delta = 1e-7
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, y_true.shape[0])
        y_pred = y_pred.reshape(1, y_pred.shape[0])
    return -np.sum(y_true * np.log(y_pred + delta)) / y_pred.shape[0]


def cross_entropy_batch_2(y_true, y_pred):
    delta = 1e-7
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, y_true.shape[0])
        y_pred = y_pred.reshape(1, y_pred.shape[0])
    return -np.sum(np.log(y_pred[:, y_true])) / y_true.shape[0]


if __name__ == "__main__":
    # 单个样本的交叉熵计算
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0])
    loss = cross_entropy(y_true, y_pred)
    print(loss)

    # 多个样本的交叉熵计算
    y_true_batch = np.array([[0, 0, 1], [0, 0, 1]])
    y_pred_batch = np.array([[0.9, 0.1, 0], [0.9, 0.1, 0]])
    loss_batch = cross_entropy_batch(y_true_batch, y_pred_batch)
    print(loss_batch)
