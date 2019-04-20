from sklearn import tree as tr
import graphviz
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin'


class TreeVizualizer():
    """
    Visual representation of a tree and the split points
    """
    def __init__(self, tree):
        self.tree = tree

    def draw_tree(self):
        dot_data = tr.export_graphviz(self.tree, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render()
