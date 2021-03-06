from sklearn.datasets import load_breast_cancer
import unittest

from src.models.linear.LogisticRegression import LogisticRegression
from src.models.trees.BinaryDecisionTree import BinaryDecisionTree
from src.PreProcessor import Standardizer
from src.Helpers import get_train_test_val_split
from src.cost_functions.trees.LogReg import LogReg as LRTrees


class LogisticRegressionTester(unittest.TestCase):

    def test_model_breast_cancer(self):
        data = load_breast_cancer()
        label = data["target"]
        data = data["data"]
        standardizer = Standardizer()

        train_test_split = get_train_test_val_split(data, label, 0.7, 0.2, 0.1)

        train_data = standardizer.preprocess_train_data(train_test_split.train_data)
        val_data = standardizer.preprocess_test_data(train_test_split.val_data)
        test_data = standardizer.preprocess_test_data(train_test_split.test_data)

        model = LogisticRegression(alpha=0.05, iterations=100, error_function="LogReg", verbose=False)
        model.train(train_data, train_test_split.train_label, val_data, train_test_split.val_label)
        predictions = model.predict(test_data)
        print("Test loss: {}".format(model.error_function.compute(predictions.reshape(predictions.shape[0],),
                                                                  train_test_split.test_label)))


class BinaryDecisionTreeClassificationTester(unittest.TestCase):

    def test_model_breast_cancer(self):
        data = load_breast_cancer()
        label = data["target"]
        data = data["data"]

        train_test_split = get_train_test_val_split(data, label, 0.7, 0.2, 0.1)

        model = BinaryDecisionTree(cost_function=LRTrees(), params={"max_depth": 2}, objective="classification")
        model.train(train_test_split.train_data, train_test_split.train_label, train_test_split.val_data,
                    train_test_split.val_label)
        predictions = model.predict(train_test_split.test_data)
        print("Test loss: {}".format(model.cost_function.compute(predictions.reshape(predictions.shape[0],),
                                                                 train_test_split.test_label)))


if __name__=="__main__":
    classific_tester = LogisticRegressionTester()
    classific_tester.test_model_breast_cancer()

    classific_tester = BinaryDecisionTreeClassificationTester()
    classific_tester.test_model_breast_cancer()