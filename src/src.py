import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from TreeBagging import TreeBagging
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin'
np.random.seed(100)

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

m = TreeBagging(x, y, n_estimators=1, set_samples=8)
m.trees[0].tree_traversal()
print(m.predict(x_test))

t = RandomForestRegressor(n_estimators=1, bootstrap=False)
t.fit(x, y)
print(t.predict(x_test))
viz = TreeVizualizer(t.estimators_[0])
viz.draw_tree()

print(x_test)