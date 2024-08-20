# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import matplotlib.pyplot as plt
import pandas as pd

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import omninet as omni
from omninet.analysis import get_tests as tests
from omninet.analysis import reduce_series
import databases as connector

# Sets up database and parameters
engine = connector.sqlalchemy_connect("model_data")
table_root = "fashion"
suptitle = "Model Epoch Wall Times on FashionMNIST"

names = [
    tests("dense1t", 11),
    tests("multi1t", 11),
    tests("hebbian2t", 11),
    tests("multi_hebbian2", 11),
    tests("memory1t", 11)
]


labels = ["DNN", "multidirectional", "hebbian", "multidir. hebbian", "nueral memory"]
colors = ["r", "g", "b", "c", "y"]

metric = "time"
titles = ["Training", "Testing"]
ax_label = "Epoch Wall Time (seconds)"

# Load model data
print("Loading model data...\n")
mean_list = []
error_list = []

train_list = omni.analysis.combine_data_series(names, engine, table_root + "_training", mean=False)
mean_list.append(reduce_series(train_list, lambda df: df.mean()))
error_list.append(reduce_series(train_list, lambda df: df.sem()))

test_list = omni.analysis.combine_data_series(names, engine, table_root + "_testing", mean=False)
test_data = pd.concat(test_list, ignore_index=True)
mean_list.append(reduce_series(test_list, lambda df: df.mean()))
error_list.append(reduce_series(test_list, lambda df: df.sem()))


# Graphs box and whisker plots for wall time
plt.rcParams.update({"font.size": 15})
fig, axes = plt.subplots(1, 2)
fig.suptitle(suptitle, fontsize=20)
for num, (title, mean, error) in enumerate(zip(titles, mean_list, error_list)):
    axes[num].set_title(title)
    axes[num].bar(labels, mean[metric], yerr=error[metric], color=colors)
    for label in axes[num].get_xticklabels(): 
        label.set_rotation(-20)
    axes[num].set_ylabel(ax_label)


plt.show()

