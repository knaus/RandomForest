import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, x_tree, y_tree):
        """
        x: data frame with independent variables to be used for tree building
        y: dependent variable
        left: decision tree containing rows with elements <= split point
        right: decision tree containing rows with elements > split point
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
        self.tree_builder()

    def is_leaf(self):
        """ If the data set passed on has only one row, you're at a leaf. Can modify to account for min_leaf
        """
        return self.split_score == float('inf')

    def tree_builder(self):
        self.split_col, split_row, self.split_score, ids_rhs, ids_lhs = self.find_best_split()
        if self.is_leaf():
            return
        else:
            self.split_row = self.x.iloc[split_row, self.split_col]
            x_right = self.x.iloc[ids_rhs, :]
            x_left = self.x.iloc[ids_lhs, :]
            y_right = self.y[ids_rhs]
            y_left = self.y[ids_lhs]
            self.right = DecisionTree(x_right, y_right)
            self.left = DecisionTree(x_left, y_left)

    def find_best_split(self):
        """The best split minimizes variance. All possible values across rows and columns are tried
        """
        split_col, split_row, split_score, ids_rhs, ids_lhs = None, None, float('inf'), None, None
        r = self.x.shape[0]
        n = self.x.shape[1]
        for row in range(r):
            for col in range(n):
                score_, rhs_, lhs_ = self.score_split(row, col)
                if score_ < split_score:
                    split_col, split_row, split_score, ids_rhs, ids_lhs = col, row, score_, rhs_, lhs_
        return split_col, split_row, split_score, ids_rhs, ids_lhs

    def score_split(self, row, col):
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

    def tree_traversal(self):
        """ Traverse a tree and print out all the leaves
        """
        if self.is_leaf():
            print(self.value)
        else:
            print('split col:', self.split_col)
            print('split row:', self.split_row)
            print('samples', self.x.shape[0])
            print('split value:', self.value)
            self.left.tree_traversal()
            self.right.tree_traversal()

    def predict(self, x_test):
        """Calls predict_row for every row of the x_test data frame
        """
        return [self.predict_row(row) for row in x_test.itertuples(index=False)]

    def predict_row(self, row):
        if self.is_leaf(): return self.value
        t = self.left if row[self.split_col] <= self.split_row else self.right
        return t.predict_row(row)

