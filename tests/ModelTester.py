import pandas as pd
import numpy as np
from unittest import TestCase
from sklearn.datasets import load_boston

from src.LinearRegression import LinearRegression
from src.PreProcessor import Standardizer


class LinearModelTester():

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


if __name__=="__main__":
    model = LinearModelTester()
    model.test_model_boston()