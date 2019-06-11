import pandas as pd
import numpy as np
import unittest
from sklearn.datasets import load_boston

from src.models.linear.LinearRegression import LinearRegression
from src.models.trees.BinaryDecisionTreeRegression import BinaryDecisionTreeRegression
from src.PreProcessor import Standardizer
from src.Helpers import get_train_test_val_split


class LinearRegressionTester(unittest.TestCase):

    def test_model_small(self):
        data = pd.read_csv("../data/regression.txt")
        data = np.array(data.values)
        standardizer = Standardizer()

        train_data = data[0:26, 0:2]
        train_data = standardizer.preprocess_train_data(train_data)
        train_label = data[0:26, 2]

        val_data = data[26:35, 0:2]
        val_data = standardizer.preprocess_test_data(val_data)
        val_label = data[26:35, 2]

        test_data = data[35:47, 0:2]
        test_data = standardizer.preprocess_test_data(test_data)

        model = LinearRegression(alpha=0.4, iterations=100, error_function="mse", verbose=True)
        model.train(train_data, train_label, val_data, val_label)
        print(model._theta)
        predictions = model.predict(test_data)
        print(predictions)

    def test_model_boston(self):
        data = load_boston()
        label = data["target"]
        data = data["data"]
        standardizer = Standardizer()

        train_test_split = get_train_test_val_split(data, label, 0.7, 0.2, 0.1)

        train_data = standardizer.preprocess_train_data(train_test_split.train_data)
        val_data = standardizer.preprocess_test_data(train_test_split.val_data)
        test_data = standardizer.preprocess_test_data(train_test_split.test_data)

        model = LinearRegression(alpha=0.05, iterations=100, error_function="mse", verbose=True)
        model.train(train_data, train_test_split.train_label, val_data, train_test_split.val_label)
        print(model._theta)
        predictions = model.predict(test_data)
        print("Test loss: {}".format(model.error_function.compute(predictions.reshape(predictions.shape[0],),
                                                                  train_test_split.test_label)))


class BinaryDecisionTreeRegressionTester(unittest.TestCase):

    def test_model_small(self):
        data = pd.read_csv("../data/regression.txt")
        data = np.array(data.values)

        train_data = data[0:26, 0:2]
        train_label = data[0:26, 2]

        val_data = data[26:35, 0:2]
        val_label = data[26:35, 2]

        test_data = data[35:47, 0:2]

        model = BinaryDecisionTreeRegression()
        model.train(train_data, train_label, val_data, val_label)

        predictions = model.predict(test_data)
        print(predictions)

    def test_model_boston(self):
        data = load_boston()
        label = data["target"]
        data = data["data"]
        standardizer = Standardizer()

        train_test_split = get_train_test_val_split(data, label, 0.7, 0.2, 0.1)

        model = BinaryDecisionTreeRegression()
        model.train(train_test_split.train_data, train_test_split.train_label,
                    train_test_split.val_data, train_test_split.val_label)
        predictions = model.predict(train_test_split.test_data)
        print("Test loss: {}".format(model.error_function.  compute(predictions.reshape(predictions.shape[0],),
                                                                  train_test_split.test_label)))
