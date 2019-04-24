import sys
sys.path.append('../src')
import unittest
import numpy as np
import pandas as pd
from src.TreeBagging import TreeBagging
from src.DecisionTree import DecisionTree

np.random.seed(100)



# print(test.isLeaf())# testing if the code for is leaf is true. made a single node and prints that it's a leaf

# what happens when sample size is bigger than y?

# get a simple data frame with one predictor and compare results with the sklearn regressor

# unit tests mean I need to test each function that it does what I think it should.

# test that teh scoreSplit function works when one side is 0.

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'YearMade': [1984, 2012, 1985, 2011, 1983, 1985, 1986, 1982],
                       'TypeOfCar': [0, 1, 0, 0, 1, 1, 1, 0],
                       'Price': [2000, 5000, 3000, 2000, 5000, 3000, 4000, 4000]},
                      index=[0, 1, 2, 3, 4, 5, 6, 7])
        x = df.iloc[0:8, 0:2]
        y = np.array(df['Price'])
        self.m = TreeBagging(x, y, n_estimators=1, set_samples=7)
        self.a = set(self.m.generate_row_index(7))
        self.b = set([i for i in range(len(y))])

    def test_generate_row_index_length(self):
        self.assertEqual(len(self.m.generate_row_index(set_samples=7)), 7)

    def test_generate_row_index_subset(self):
        self.assertEqual(self.a - self.b, set())


if __name__ == '__main__':
    unittest.main()





