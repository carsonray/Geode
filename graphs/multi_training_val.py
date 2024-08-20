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

import omninet as omni
from omninet.analysis import get_tests as tests
import databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
suptitle = "Model Training Validation Performance with Task Knowledge Comparison"
table_root = "image_tasks2"

names = []
names.append([
    tests("taskconv2t", 5, use_root=False),
    tests("multiconv1t", 5, use_root=False),
    tests("hebbconv1", 5, use_root=False),
    tests("multihebbconv2", 5, use_root=False)
])
names.append([
    tests("taskconv1t", 5, use_root=False),
    tests("multiconv1n", 5, use_root=False),
    tests("hebbconv1n", 5, use_root=False),
    tests("multihebbconv2n", 5, use_root=False)
])

labels = ["convolutional", "multidir. conv.", "hebbian conv.", "multidir. hebb. conv."]
colors = ["purple", "orangered", "navy", "mediumslateblue"]

metric_titles = ["Loss", "Accuracy"]
metrics = ["loss", "accuracy"]
ax_labels = ["Loss (mean sqaured error)", "Accuracy"]

# Load model data
print("Loading model data...\n")

train_list = []
test_list = []
for i in range(2):
    train_list.append(omni.analysis.combine_data_series(names[i], engine, table_root + "_training", mean_col="epoch"))

    test_list.append(omni.analysis.combine_data_series(names[i], engine, table_root + "_testing"))


print("Displaying comparisons...\n")
plt.rcParams.update({"font.size": 15})
fig, axes = omni.analysis.get_train_fig(suptitle, metric_titles, ax_labels)

omni.analysis.plot_train_series(fig, axes, train_list[0], test_list[0], labels, metrics, colors=colors, use="val", legend_loc=(0.34, 0.01))

omni.analysis.plot_train_series(fig, axes, train_list[1], test_list[1], labels, metrics, colors=colors, use="val",
                                linestyle="dotted", legend=False)

for num, metric in enumerate(metrics):
    for data_num, color in enumerate(colors):
        axes[num].fill_between(train_list[0][data_num]["epoch"], train_list[0][data_num]["val_" + metric],
                                        train_list[1][data_num]["val_" + metric], color=color, alpha=0.2)

proxies = [
    plt.plot([], color="black", linestyle="solid", marker="v", label="known task"),
    plt.plot([], color="black", linestyle="dotted", marker="v", label="unknown task"),
    plt.plot([], color="black", linestyle="dashed", label="test")
]

proxies = [proxy[0] for proxy in proxies]
proxy_legend = axes[1].legend(handles=proxies, loc=(0.7, 0.07))
axes[1].add_artist(proxy_legend)

plt.show()

