import abc
import numpy as np


class PreProcessor(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def preprocess_train_data(self, train_data: np.ndarray):
        pass

    @abc.abstractmethod
    def preprocess_test_data(self, data: np.ndarray):
        pass

    @abc.abstractmethod
    def revert_preprocessing(self, data: np.ndarray):
        pass


class Standardizer(PreProcessor):

    def __init__(self):
        self._mu = None
        self._sd = None
        super().__init__()

    def preprocess_train_data(self, train_data: np.ndarray):
        self._mu = np.mean(train_data, axis=0).reshape((1, train_data.shape[1]))
        self._sd = np.std(train_data, axis=0).reshape((1, train_data.shape[1]))
        train_data = train_data.copy()

        return (train_data - self._mu) / self._sd

    def preprocess_test_data(self, data: np.ndarray):
        assert self._mu is not None, "Preprocessor must be trained first!"

        data = data.copy()
        return (data - self._mu) / self._sd

    def revert_preprocessing(self, data: np.ndarray):
        assert self._mu is not None, "Preprocessor must be trained first!"

        data = data.copy()
        return data * self._sd + self._mu


class OneHotEncoder(PreProcessor):

    def __init__(self):
        super().__init__()

    def preprocess_train_data(self, train_data: np.ndarray):
        pass

    def preprocess_test_data(self, data: np.ndarray):
        pass

    def revert_preprocessing(self, data: np.ndarray):
        pass
