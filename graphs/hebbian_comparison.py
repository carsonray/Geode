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
names = ["dense1", "dense2"]
colors = ["r", "g"]
metrics = ["loss", "accuracy"]
table_root = "fashion"
labels = ["Loss (mean sqaured error)", "Accuracy"]
titles = ["Loss", "Accuracy"]

# Load model data
print("Loading model data...\n")

raw_train_data = geode.combine_data_handlers(names, 
                                            db_table=f"{table_root}_training",
                                            database=engine)

raw_test_data = geode.combine_data_handlers(names, 
                                            db_table=f"{table_root}_testing",
                                            database=engine)

print("Displaying comparisons...\n")

fig, axes = plt.subplots(1, len(metrics))
fig.suptitle("Hebbian Model Training Performance on  FashionMNIST")
for title, label, (num, metric) in zip(titles, labels, enumerate(metrics)):
    for name, color in zip(names, colors):
        train_data = raw_train_data[raw_train_data["name"] == name]
        test_data = raw_test_data[raw_test_data["name"] == name]
        axes[num].plot(train_data["epoch"], train_data[metric], f"{color}-", 
                    label=f"{name}_train")
        axes[num].plot(train_data["epoch"], train_data[f"val_{metric}"], f"{color}-", 
                    marker='+',
                    label=f"{name}_validation")
        axes[num].fill_between(train_data["epoch"], train_data[metric], train_data[f"val_{metric}"], color=color, alpha=0.2)
        axes[num].hlines(test_data[metric], train_data["epoch"].min(), train_data["epoch"].max(), 
                    colors=color,
                    linestyles='dashed',
                    label=f"{name}_test")
        axes[num].set_ylabel(label)
        axes[num].set_xlabel("Training Epoch")
        axes[num].set_xticks(train_data["epoch"])

    axes[num].set_title(title)
    axes[num].legend()


plt.show()