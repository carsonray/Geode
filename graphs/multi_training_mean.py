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
import databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
suptitle = "Model Training Comparison on 3 Combined Image Tasks"
table_root = "image_tasks2"

names = [
    tests("taskconv2t", 5, use_root=False),
    tests("multiconv1t", 5, use_root=False),
    tests("hebbconv1", 5, use_root=False),
    tests("multihebbconv2", 5, use_root=False)
]

labels = ["convolutional", "multidir. conv.", "hebbian conv.", "multidir. hebb. conv."]
colors = ["purple", "orangered", "navy", "mediumslateblue"]

metric_titles = ["Loss", "Accuracy"]
metrics = ["loss", "accuracy"]
ax_labels = ["Loss (mean sqaured error)", "Accuracy"]

# Load model data
print("Loading model data...\n")

train_list = geode.analysis.combine_data_series(names, engine, table_root + "_training", mean_col="epoch")

test_list = geode.analysis.combine_data_series(names, engine, table_root + "_testing")

print("Displaying comparisons...\n")
plt.rcParams.update({"font.size": 15})
fig, axes = geode.analysis.get_train_fig(suptitle, metric_titles, ax_labels)
geode.analysis.plot_train_series(fig, axes, train_list, test_list, labels, metrics, colors=colors, legend_loc="lower center")
geode.analysis.train_proxy_legend(axes)
plt.show()



