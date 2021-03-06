import pandas as pd
import numpy as np
import unittest
from sklearn.datasets import load_boston

from src.models.linear.LinearRegression import LinearRegression
from src.models.trees.BinaryDecisionTree import BinaryDecisionTree
from src.models.networks.NeuralNetwork import NeuralNetwork
from src.cost_functions.trees.MeanSquaredError import MeanSqaredError as MSETree
from src.cost_functions.network.MeanSquaredError import MeanSqaredError as MSENetwork
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
        test_label = data[35:47, 2]

        model = LinearRegression(alpha=0.4, iterations=100, cost_function="mse", verbose=False)
        model.train(train_data, train_label, val_data, val_label)
        predictions = model.predict(test_data)
        print("Test loss: {}".format(model.cost_function.compute(predictions.reshape(predictions.shape[0],),
                                                                 test_label)))

    def test_model_boston(self):
        data = load_boston()
        label = data["target"]
        data = data["data"]
        standardizer = Standardizer()

        train_test_split = get_train_test_val_split(data, label, 0.7, 0.2, 0.1)

        train_data = standardizer.preprocess_train_data(train_test_split.train_data)
        val_data = standardizer.preprocess_test_data(train_test_split.val_data)
        test_data = standardizer.preprocess_test_data(train_test_split.test_data)

        model = LinearRegression(alpha=0.05, iterations=100, cost_function="mse", verbose=False)
        model.train(train_data, train_test_split.train_label, val_data, train_test_split.val_label)
        predictions = model.predict(test_data)
        print("Test loss: {}".format(model.cost_function.compute(predictions.reshape(predictions.shape[0],),
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
        test_label = data[35:47, 2]
        params = {"max_depth": 3, "save_path_tree_struct": "../src/graphs/structure.json"}
        model = BinaryDecisionTree(cost_function=MSETree(), params=params, objective="regression")
        model.train(train_data, train_label, val_data, val_label)

        predictions = model.predict(test_data)
        print("Test loss: {}".format(model.cost_function.  compute(predictions.reshape(predictions.shape[0],),
                                                                   test_label)))

    def test_model_boston(self):
        data = load_boston()
        label = data["target"]
        data = data["data"]

        train_test_split = get_train_test_val_split(data, label, 0.7, 0.2, 0.1)
        params = {"max_depth": 3, "save_path_tree_struct": "../src/graphs/structure.json"}

        model = BinaryDecisionTree(cost_function=MSETree(), params=params, objective="regression")
        model.train(train_test_split.train_data, train_test_split.train_label,
                    train_test_split.val_data, train_test_split.val_label)
        predictions = model.predict(train_test_split.test_data)
        print("Test loss: {}".format(model.cost_function.  compute(predictions.reshape(predictions.shape[0],),
                                                                   train_test_split.test_label)))


class NeuralNetworkRegressionTester(unittest.TestCase):

    def test_model_small(self):
        data = pd.read_csv("../data/regression.txt")
        data = np.array(data.values)
        standardizer = Standardizer()
        standardizer_label = Standardizer()

        train_data = data[0:26, 0:2]
        train_data = standardizer.preprocess_train_data(train_data)
        train_label = data[0:26, 2]
        train_label = standardizer_label.preprocess_train_data(train_label.reshape(train_label.shape[0], 1))

        val_data = data[26:35, 0:2]
        val_data = standardizer.preprocess_test_data(val_data)
        val_label = data[26:35, 2]
        val_label = standardizer_label.preprocess_test_data(val_label.reshape(val_label.shape[0], 1))

        test_data = data[35:47, 0:2]
        test_label = data[35:47, 2]
        test_data = standardizer.preprocess_test_data(test_data)
        params = {"iterations": 10}
        model = NeuralNetwork(cost_function=MSENetwork(), input_shape=(2,), params=params,
                              neurons=[4, 8, 1], activation_functions=["sigmoid", "sigmoid",  "linear"], epochs=100,
                              learning_rate=0.0001, verbose=False)
        model.train(train_data, train_label, val_data, val_label)

        predictions = model.predict(test_data)
        predictions = standardizer_label.revert_preprocessing(predictions)
        print("Test loss: {}".format(model.cost_function.compute(predictions.reshape(predictions.shape[0],),
                                                                 test_label, aggregation=True)))

    def test_model_boston(self):
        data = load_boston()
        label = data["target"]
        data = data["data"]
        standardizer = Standardizer()
        standardizer_label = Standardizer()

        train_test_split = get_train_test_val_split(data, label, 0.6, 0.2, 0.2)
        train_data = standardizer.preprocess_train_data(train_test_split.train_data)
        val_data = standardizer.preprocess_test_data(train_test_split.val_data)
        test_data = standardizer.preprocess_test_data(train_test_split.test_data)

        train_label = standardizer_label.preprocess_train_data(train_test_split.train_label.
                                                               reshape(train_test_split.train_label.shape[0], 1))
        val_label = standardizer_label.preprocess_test_data(train_test_split.val_label.
                                                            reshape(train_test_split.val_label.shape[0], 1))


        model = NeuralNetwork(cost_function=MSENetwork(), params={}, input_shape=(13,), epochs=100,
                              neurons=[4, 8, 1], activation_functions=["sigmoid", "sigmoid", "linear"],
                              learning_rate=0.0001, verbose=False)
        model.train(train_data, train_label, val_data, val_label)
        predictions = model.predict(test_data)
        predictions = standardizer_label.revert_preprocessing(predictions)
        print("Test loss: {}".format(np.sum(model.cost_function.compute(predictions.reshape(predictions.shape[0],),
                                                                        train_test_split.test_label, aggregation=True))))


if __name__=="__main__":
    linear = LinearRegressionTester()
    linear.test_model_boston()
    linear.test_model_small()

    tree = BinaryDecisionTreeRegressionTester()
    tree.test_model_small()
    tree.test_model_boston()

    tree = NeuralNetworkRegressionTester()
    tree.test_model_small()
    tree.test_model_boston()


