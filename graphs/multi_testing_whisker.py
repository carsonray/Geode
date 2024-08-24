# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import geode
from geode.analysis import get_tests as tests
import databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
suptitle = "Model Test Distribution on 3 Combined Image Tasks"
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

test_list = []
for i in range(2):
    test_list.append(geode.analysis.combine_data_series(names[i], engine, table_root + "_testing", mean=False))

print("Displaying comparisons...\n")
plt.rcParams.update({"font.size": 15})
fig, axes = geode.analysis.get_test_fig(suptitle, metric_titles, ax_labels)

x = np.arange(len(labels))
width = 0.4
for num, (title, ax_label, metric) in enumerate(zip(metric_titles, ax_labels, metrics)):
    metric_data = [[data[metric] for data in test] for test in test_list]
    axes[num].set_title(title)
    box1 = axes[num].boxplot(metric_data[0], positions=(x - width/2), patch_artist=True, showmeans=True, labels=labels, widths=width)
    box2 = axes[num].boxplot(metric_data[1], positions=(x + width/2), patch_artist=True, showmeans=True, labels=labels, widths=width)

    axes[num].set_xticks(x, labels, rotation=-20)
    axes[num].set_ylabel(ax_label)

    # Sets colors
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)

    for patch, color in zip(box2['boxes'], colors):
        patch.set(fill=False, edgecolor=color)

    # Creates legend
    proxies = [
        patches.Patch(color="black", label="known task"),
        patches.Patch(color="black", fill=False, label="unknown task"),
    ]

    axes[num].legend(handles=proxies, loc="upper center")

plt.show()

