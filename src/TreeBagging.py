import numpy as np
import pandas as pd
from DecisionTree import DecisionTree

np.random.seed(100)

class TreeBagging:
    def __init__(self, x, y, n_estimators=1, set_samples=7):
        self.trees = []
        self.x, self.y = x, y
        self.n_estimators, self.set_samples = n_estimators, set_samples
        for i in range(n_estimators):
            ids = self.generateRowIndex(self.set_samples)
            x_tree = x.iloc[ids, :]
            y_tree = y[ids]
            self.trees.append(DecisionTree(x_tree, y_tree))

    def generateRowIndex(self, set_samples):
        """ Pick a set_samples number of indices without replacement from the rows available in x to build the tree
        """
        rnd_ids = np.random.permutation(len(self.y))[:self.set_samples]
        return rnd_ids

    def fixColumns(self, x_test):
        """ Reorder x_test that we use for predictions to match the column order of x, the data frame used for training
        """
        cols = self.x.columns.tolist()
        return x_test[cols]

    def predict(self, x_test):
        x_test = self.fixColumns(x_test)
        return np.mean([t.predict(x_test) for t in self.trees], axis=0)


df = pd.DataFrame({'YearMade': [1984, 2012, 1985, 2011, 1983, 1985, 1986, 1982],
                       'TypeOfCar': [0, 1, 0, 0, 1, 1, 1, 0],
                       'Price': [2000, 5000, 3000, 2000, 5000, 3000, 4000, 4000]},
                      index=[0, 1, 2, 3, 4, 5, 6, 7])
x = df.iloc[0:8, 0:2]
y = np.array(df['Price'])
m = TreeBagging(x, y, n_estimators=1, set_samples=7)
print(m.generateRowIndex(7))