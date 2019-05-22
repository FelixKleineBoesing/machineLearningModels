from typing import Union
import numpy as np
import pandas as pd


def get_train_test_val_split(data: Union[np.ndarray, pd.DataFrame], label: Union[np.ndarray, pd.DataFrame, list],
                             train_share:float=0.8, test_share: float=0.2, val_share: float=None):
    n = data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[:int(train_share  * n)]
    if val_share is not None:
        test_indices = indices[int(train_share * n):int((train_share + test_share) * n)]
        val_indices = indices[int((train_share+test_share)*n):]
        val_data = data[val_indices, :]
        val_label = label[val_indices]
    else:
        test_indices = indices[int(train_share * n):]
        val_data = None
        val_label = None

    train_data = data[train_indices, :]
    train_label = label[train_indices]
    test_data = data[test_indices, :]
    test_label = label[test_indices]

    return TrainTestSplit(train_data, train_label, test_data, test_label, val_data, val_label)


class TrainTestSplit:

    def __init__(self, train_data, train_label, test_data, test_label, val_data=None, val_label=None):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.val_data = val_data
        self.val_label = val_label


def sigmoid(z):
    return 1/ (1 + np.exp(-z))