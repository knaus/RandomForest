import unittest
import pandas as pd
import numpy as np
from src.src import *
from sklearn.ensemble import RandomForestRegressor
from src.TreeVizualizer import TreeVizualizer

# print(test.isLeaf())# testing if the code for is leaf is true. made a single node and prints that it's a leaf

# what happens when sample size is bigger than y?

# get a simple data frame with one predictor and compare results with the sklearn regressor

# unit tests mean I need to test each function that it does what I think it should.

# test that teh scoreSplit function works when one side is 0.

df = pd.DataFrame({'YearMade': [1984, 2012, 1985, 2011, 1983, 1985, 1986, 1982],
                   'TypeOfCar': [0, 1, 0, 0, 1, 1, 1, 0],
                   'Price': [2000, 5000, 3000, 2000, 5000, 3000, 4000, 4000]},
                  index=[0, 1, 2, 3, 4, 5, 6, 7])
x = df.iloc[0:7, 0:2]
y = np.array(df['Price'])



df = pd.DataFrame({'YearMade': [1984, 2012, 1985, 2011, 1983, 1985, 1986, 1982],
                   'TypeOfCar': [0, 1, 0, 0, 1, 1, 1, 0],
                   'Price': [2000, 5000, 3000, 2000, 5000, 3000, 4000, 4000]},
                  index=[0, 1, 2, 3, 4, 5, 6, 7])
x = df.iloc[0:8, 0:2]
y = np.array(df['Price'])

df_test = pd.DataFrame({'YearMade': [1985, 2012, 1986, 2012, 1983, 1987, 1986, 1982],
                        'TypeOfCar': [0, 1, 0, 0, 1, 0, 1, 1],
                        'Price': [2000, 6000, 2000, 2000, 6000, 4000, 4000, 4000]},
                       index=[0, 1, 2, 3, 4, 5, 6, 7])
x_test = df_test.iloc[0:8, 0:2]
m = TreeBagging(x, y, n_estimators=1, set_samples=8, max_features=0.5)
m.trees[0].treeTraversal()

print(m.predict(x_test))

t = RandomForestRegressor(n_estimators=1, bootstrap=False)
t.fit(x, y)
print(t.predict(x_test))
#
viz = TreeVizualizer(t.estimators_[0])
viz.draw_tree()
