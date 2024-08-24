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
import databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
table_root = "image_tasks"
suptitle = "Model Test Performance on 3 Combined Image Tasks"
names = ["conv1", "conv2", "multi1"]
labels = ["convolutional model", "task conv. model", "multidirectional conv. model"]
colors = ["purple", "orangered", "midnightblue"]

metrics = ["mse", "accuracy"]
ax_labels = ["Loss (mean sqaured error)", "Accuracy"]
titles = ["Loss", "Accuracy"]

# Load model data
print("Loading model data...\n")

test_data = geode.combine_data_handlers(names, 
                                        db_table=f"{table_root}_testing",
                                        database=engine)


print("Displaying comparisons...\n")

fig, axes = plt.subplots(1, len(metrics))
fig.suptitle(suptitle, fontsize=15)
for num, (title, ax_label, metric) in enumerate(zip(titles, ax_labels, metrics)):
    axes[num].set_title(title)
    axes[num].bar(labels, test_data[metric], color=colors)
    axes[num].set_ylabel(ax_label)

plt.show()
