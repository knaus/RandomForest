import unittest
import numpy as np
import pandas as pd
import sys

sys.path.append('../src')

from src.DecisionTree import DecisionTree

np.random.seed(100)


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'YearMade': [1985, 1986, 2000],
                           'Price': [2000, 3000, 4000]},
                          index=[0, 1, 2])
        self.x = df.iloc[:, 0:1]
        self.y = np.array(df['Price'])

    def test_is_leaf(self):
        tree = DecisionTree(self.x, self.y)
        self.assertEqual(tree.left.is_leaf(), True)

    def test_find_best_split(self):
        tree = DecisionTree(self.x, self.y)
        self.assertEqual(tree.split_row, 1985)

    def test_score_split(self):
        tree = DecisionTree(self.x, self.y)
        score, _, _ = tree.score_split(1, 0)
        self.assertEqual(score, 1000)

    def test_predict(self):
        df = pd.DataFrame({'YearMade': [1984, 1986, 2000, 1984, 1986, 2000, 1984, 1986],
                           'Price': [2000, 2000, 4000, 2000, 2000, 4000, 2000, 2000]},
                          index=[0, 1, 2, 3, 4, 5, 6, 7])
        x = df.iloc[:, 0:1]
        y = np.array(df['Price'])
        tree = DecisionTree(x, y)
        x_test = pd.DataFrame({'YearMade': [1985, 1985, 2002]},
                              index=[0, 1, 2])
        test_predictions = tree.predict(x_test)
        expected_predictions = [2000, 2000, 4000]
        self.assertEqual((np.array(test_predictions) == np.array(expected_predictions)).all(), True)
