import sys
from PyQt5.QtWidgets import QApplication
from sklearn import datasets
import pandas as pd

from classifiers import mlp_test_and_train, random_forest_test_and_train, random_forest_learning_curve
from plots import draw_features
from ui import Example
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # ex = Example()
    # sys.exit(app.exec_())
    random_forest_test_and_train('datasets/rice_with_spreading.xlsx', 'datasets/rice_withOUT_spreading.xlsx', ["Ширина", "Высота", "Площадь"])
    # draw_features('datasets/plots/mixed.xlsx', "Square", "Length")
    # random_forest_learning_curve('datasets/normalized_rice.xlsx', ["Ширина", "Высота", "Площадь", "Периметр"])