import sys

sys.path.append('../src')
import unittest
import numpy as np
import pandas as pd
import collections
from src.TreeBagging import TreeBagging

np.random.seed(100)


class TestTreeBagging(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'YearMade': [1984, 2012, 1985, 2011, 1983, 1985, 1986, 1982],
                           'TypeOfCar': [0, 1, 0, 0, 1, 1, 1, 0],
                           'Price': [2000, 5000, 3000, 2000, 5000, 3000, 4000, 4000]},
                          index=[0, 1, 2, 3, 4, 5, 6, 7])
        self.x = df.iloc[0:8, 0:2]
        self.y = np.array(df['Price'])
        self.m = TreeBagging(self.x, self.y, n_estimators=1, set_samples=7)

    def test_generate_row_index_length(self):
        self.assertEqual(len(self.m.generate_row_index(set_samples=7)), 7)

    def test_generate_row_index_subset(self):
        a = set(self.m.generate_row_index(7))
        b = set([i for i in range(len(self.y))])
        self.assertEqual(a - b, set())

    def test_fix_columns(self):
        cols_x_test = ['TypeOfCar', 'YearMade']
        x_test = self.x[cols_x_test]
        cols_x = self.x.columns.tolist()
        x_test = self.m.fix_columns(x_test)
        cols_x_test = x_test.columns.tolist()
        compare = [i == j for i, j in zip(cols_x, cols_x_test)]
        self.assertTrue(compare)

    def test_predict(self):
        df = pd.DataFrame({'YearMade': [1984, 1986, 2000, 1984, 1986, 2000, 1984, 1986],
                           'Price': [2000, 2000, 4000, 2000, 2000, 4000, 2000, 2000]},
                          index=[0, 1, 2, 3, 4, 5, 6, 7])
        self.x = df.iloc[:, 0:1]
        self.y = np.array(df['Price'])
        self.m = TreeBagging(self.x, self.y, n_estimators=1, set_samples=7)
        x_test = pd.DataFrame({'YearMade': [1985, 1985, 2002]},
                              index=[0, 1, 2])
        test_predictions = self.m.predict(x_test)
        my_predictions = [2000, 2000, 4000]
        self.assertEqual(collections.Counter(test_predictions), collections.Counter(my_predictions))


if __name__ == '__main__':
    unittest.main()
