# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import geode
from geode.analysis import get_tests as tests
import geode.databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
table_root = "fashion"
suptitle = "Model Training Performance on FashionMNIST"

names = [
    (tests("dense1", 11) + tests("dense1t", 11)),
    (tests("multi1", 11) + tests("multi1t", 11)),
    (tests("hebbian2", 11) + tests("hebbian2t", 11)),
    tests("multi_hebbian2", 22),
    (tests("memory1", 11) + tests("memory1t", 11))
]
labels = ["DNN", "multidirectional", "hebbian", "multidirectional hebbian", "nueral memory"]
colors = ["r", "g", "b", "c", "y"]

metric_titles = ["Loss", "Accuracy"]
metrics = ["loss", "accuracy"]
ax_labels = ["Loss (categorical crossentropy)", "Accuracy"]

# Load model data
print("Loading model data...\n")

train_list = geode.analysis.combine_data_series(names, engine, table_root + "_training", mean_col="epoch")

test_list = geode.analysis.combine_data_series(names, engine, table_root + "_testing")

print("Displaying comparisons...\n")
plt.rcParams.update({"font.size": 15})
fig, axes = geode.analysis.get_train_fig(suptitle, metric_titles, ax_labels)
geode.analysis.plot_train_series(fig, axes, train_list, test_list, labels, metrics, colors=colors)
geode.analysis.train_proxy_legend(axes)
plt.show()
