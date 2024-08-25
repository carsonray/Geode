# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import geode
import geode.databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
table_root = "fashion"
suptitle = "Model Test Performance on FashionMNIST"

name_roots = ["dense1", "multi1", "hebbian2", "memory1"]
labels = ["DNN", "multidirectional model", "hebbian model", "memory model"]
colors = ["r", "g", "b", "y"]

metrics = ["loss", "accuracy"]
ax_labels = ["Loss (categorical crossentropy)", "Accuracy"]
titles = ["Loss", "Accuracy"]

# Load model data
print("Loading model data...\n")

test_means = pd.DataFrame(columns=metrics)
test_error = pd.DataFrame(columns=metrics)
for root in name_roots:
    # Compiles names
    names = [root]
    num = 10
    for i in range(num):
        names.append(f"{root}-{i}")

    raw = geode.combine_data_handlers(names, 
                                    db_table=f"{table_root}_testing",
                                    database=engine)
    raw = raw.drop(["name", "epoch"], axis=1)
    
    # Gets mean of data
    test_means = test_means.append(raw.mean(axis=0), ignore_index=True)
    
    # Gets standard errors of data
    test_error = test_error.append(raw.sem(axis=0), ignore_index=True)

print(test_means)
print(test_error)

print("Displaying comparisons...\n")

fig, axes = plt.subplots(1, len(metrics))
fig.suptitle(suptitle, fontsize=15)
for num, (title, ax_label, metric) in enumerate(zip(titles, ax_labels, metrics)):
    axes[num].set_title(title)
    axes[num].bar(labels, test_means[metric], yerr=test_error[metric], color=colors)
    axes[num].set_ylabel(ax_label)

plt.show()
