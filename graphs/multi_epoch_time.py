# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import omninet as omni
from omninet.analysis import get_tests as tests
from omninet.analysis import reduce_series
import databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
suptitle = "Model Epoch Wall Times on 3 Combined Image Tasks"
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

metric = "time"
titles = ["Training", "Testing"]
ax_label = "Epoch Wall Time (seconds)"

# Load model data
print("Loading model data...\n")
mean_list = []
error_list = []

for num, extension in enumerate(("_training", "_testing")):
    mean_list.append([])
    error_list.append([])
    
    for i in range(2):
        data_list = omni.analysis.combine_data_series(names[i], engine, table_root + extension, mean=False)

        mean_list[num].append(reduce_series(data_list, lambda df: df.mean()))
        error_list[num].append(reduce_series(data_list, lambda df: df.sem()))


# Graphs box and whisker plots for wall time
plt.rcParams.update({"font.size": 15})
fig, axes = plt.subplots(1, 2)
fig.suptitle(suptitle, fontsize=20)

width = 0.25
x = np.arange(len(labels))

for num, (title, mean, error) in enumerate(zip(titles, mean_list, error_list)):
    axes[num].set_title(title)
    axes[num].bar(x - width/2, mean[0][metric], width=width, yerr=error[0][metric], color=colors, label="known task")
    axes[num].bar(x + width/2, mean[1][metric], width=width, yerr=error[1][metric], fill=False, edgecolor=colors, label="unknown task")

    axes[num].set_xticks(x, labels)
    for label in axes[num].get_xticklabels(): 
        label.set_rotation(-20)
    axes[num].set_ylabel(ax_label)

    proxies = [
        patches.Patch(color="black", label="known task"),
        patches.Patch(color="black", fill=False, label="unknown task"),
    ]

    axes[num].legend(handles=proxies, loc=(0.43, 0.9))


plt.show()

