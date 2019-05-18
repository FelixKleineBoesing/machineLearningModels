import pandas as pd
import numpy as np
from unittest import TestCase
from src.LinearRegression import LinearRegression


class LinearModelTester(TestCase):

    def test_model(self):
        data = pd.read_csv("../data/regression.txt")
        data = np.array(data.values)
        train_data = data[0:26, 0:2]
        train_label = data[0:26, 2]
        val_data = data[26:35, 0:2]
        val_label = data[26:35, 2]
        test_data = data[35:47, 0:2]

        model = LinearRegression(alpha=0.4, iterations=10, error_function="mse", verbose=True)
        model.train(train_data, train_label, val_data, val_label)
        model.predict(test_data)


