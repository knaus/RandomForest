import numpy as np
import os


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin'
np.random.seed(100)


class TreeBagging:
    def __init__(self, x, y, n_estimators=1, set_samples=7, max_features=0.5):
        self.trees = []
        self.x, self.y = x, y
        self.n_estimators, self.set_samples, self.max_features = n_estimators, set_samples, max_features
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


class DecisionTree:
    def __init__(self, x_tree, y_tree):
        """
        x: data frame with independent variables to be used for tree building
        y: dependent variable
        left: decision tree containing rows with elements > split point
        right: decision tree containing rows with elements <= split point
        value: the predicted value for the dependent variable for the rows in the tree
        split_score: weighted standard deviation of the predicted y values for the rows in the tree. Used to identify the best split point
        split_col: the column in x where the split happens
        split_row: the value of predictor in column split_col which is used as a split point
        """
        self.x, self.y = x_tree, y_tree
        self.left = None
        self.right = None
        self.value = np.mean(self.y)
        self.split_score = float('inf')
        self.split_col = None
        self.split_row = None
        self.treeBuilder()

    def isLeaf(self):
        """ If the data set passed on has only one row, you're at a leaf. Can modify to account for min_leaf
        """
        return self.split_score == float('inf')

    def treeBuilder(self):
        self.split_col, split_row, self.split_score, ids_rhs, ids_lhs = self.findBestSplit()
        if self.isLeaf():
            return
        else:
            self.split_row = self.x.iloc[split_row, self.split_col]
            x_right = self.x.iloc[ids_rhs, :]
            x_left = self.x.iloc[ids_lhs, :]
            y_right = self.y[ids_rhs]
            y_left = self.y[ids_lhs]
            self.right = DecisionTree(x_right, y_right)
            self.left = DecisionTree(x_left, y_left)

    def findBestSplit(self):
        """The best split minimizes variance. All possible values across rows and columns are tried
        """
        split_col, split_row, split_score, ids_rhs, ids_lhs = None, None, float('inf'), None, None
        r = self.x.shape[0]
        n = self.x.shape[1]
        for row in range(r):
            for col in range(n):
                score_, rhs_, lhs_ = self.scoreSplit(row, col)
                if score_ < split_score:
                    split_col, split_row, split_score, ids_rhs, ids_lhs = col, row, score_, rhs_, lhs_
        return split_col, split_row, split_score, ids_rhs, ids_lhs

    def scoreSplit(self, row, col):
        """ The score is a weighted average of the standard deviation of the group of rows with y > split and the rows with values y <= split
        Return: the score of the split identified by row and col
        """
        split = self.x.iloc[row, col]
        ids_rhs = [elem > split for elem in self.x.iloc[:, col]]
        ids_lhs = [elem <= split for elem in self.x.iloc[:, col]]
        if sum(ids_rhs) == 0 or sum(ids_lhs) == 0: return float('inf'), ids_rhs, ids_lhs
        y_rhs = self.y[ids_rhs]
        y_lhs = self.y[ids_lhs]
        std_rhs = np.std(np.array(y_rhs))
        std_lhs = np.std(np.array(y_lhs))
        score = std_rhs * sum(ids_rhs) + std_lhs * sum(ids_lhs)
        return score, ids_rhs, ids_lhs

    def treeTraversal(self):
        """ Traverse a tree and print out all the leaves
        """
        if self.isLeaf():
            print(self.value)
        else:
            print('split col:', self.split_col)
            print('split row:', self.split_row)
            print('samples', self.x.shape[0])
            print('split value:', self.value)
            self.left.treeTraversal()
            self.right.treeTraversal()

    def predict(self, x_test):
        """Calls a predictRow for every row of the x_test data frame"""
        return [self.predictRow(row) for row in x_test.itertuples(index=False)]

    def predictRow(self, row):
        if self.isLeaf(): return self.value
        print(row)
        print('row[self.split_col] > split_row', row[self.split_col] > self.split_row)
        print('row[self.split_col]', row[self.split_col])
        print('self.split_row', self.split_row)

        t = self.left if row[self.split_col] <= self.split_row else self.right
        print(self.split_row)
        print(self.value)
        return t.predictRow(row)

