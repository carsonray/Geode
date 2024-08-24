# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import geode
from geode.analysis import get_tests as tests
import databases as connector

# Sets up database and parameters
engine = connector.sqlalchemy_connect("model_data")
table_root = "fashion"
suptitle = "Model Test Distribution on FashionMNIST"

names = [
    (tests("dense1", 11) + tests("dense1t", 11)),
    (tests("multi1", 11) + tests("multi1t", 11)),
    (tests("hebbian2", 11) + tests("hebbian2t", 11)),
    tests("multi_hebbian2", 22),
    (tests("memory1", 11) + tests("memory1t", 11))
]

pick = [0, 1, 3]

labels = ["DNN", "multidirectional", "hebbian", "multidir. hebbian", "nueral memory"]
colors = ["r", "g", "b", "c", "y"]
"""
names = [names[i] for i in pick]
labels = [labels[i] for i in pick]
colors = [colors[i] for i in pick]
"""
metric_titles = ["Loss", "Accuracy"]
metrics = ["loss", "accuracy"]
ax_labels = ["Loss (categorical crossentropy)", "Accuracy"]

# Load model data
print("Loading model data...\n")

test_list = geode.analysis.combine_data_series(names, engine, table_root + "_testing", mean=False)


# Graphs box and whisker plots for loss and accuracy
plt.rcParams.update({"font.size": 15})
geode.analysis.plot_test_whisker(test_list, labels, metrics, metric_titles=metric_titles,
                                ax_labels=ax_labels, suptitle=suptitle, colors=colors)


plt.show()

