# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import omninet as omni
from omninet.analysis import get_tests as tests
import databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
table_root = "image_tasks2"

names = [
    [
        tests("taskconv2t", 5, use_root=False),
        tests("multiconv1t", 5, use_root=False),
        tests("hebbconv1", 5, use_root=False),
        tests("multihebbconv2", 5, use_root=False),
    ],
    [
        tests("taskconv1t", 5, use_root=False),
        tests("multiconv1n", 5, use_root=False),
        tests("hebbconv1n", 5, use_root=False),
        tests("multihebbconv2n", 5, use_root=False)
    ]
]

labels = [
    "taskconv",
    "multiconv1",
    "hebbconv1",
    "multihebbconv2"
]

# Load model data
print("Loading model data...\n")

train_list = []
test_list = []
for i in range(2):
    train_list.append(omni.analysis.combine_data_series(names[i], engine, table_root + "_training", mean=False))

    test_list.append(omni.analysis.combine_data_series(names[i], engine, table_root + "_testing", mean=False))

train_list = [(one, two) for one, two in zip(*train_list)]
test_list = [(one, two) for one, two in zip(*test_list)]

# Creates blank dataframe
differential = pd.DataFrame()

for train, test, label in zip(train_list, test_list, labels):
    components = []
    for i in range(2):
        component = test[i].drop("time", axis=1)
        last_train =  train[i][train[i]["epoch"] == 19.0]
        component["loss_overfit"] = np.abs(last_train["loss"].to_numpy() - test[i]["loss"].to_numpy())
        component["accuracy_overfit"] = np.abs(last_train["accuracy"].to_numpy() - test[i]["accuracy"].to_numpy())

        components.append(component)

    # Subtracts known from unknown
    diff_point = components[1] - components[0]
    
    # Computes means and standard errors of columns
    mean_diff_point = pd.DataFrame()
    for col in diff_point.columns:
        mean_diff_point[col] = [diff_point[col].mean()]
        mean_diff_point[col + "_se"] = [diff_point[col].sem()]
    
    mean_diff_point["name"] = [label]
    
    differential = differential.append(mean_diff_point, ignore_index=True)

print(differential)
differential.to_csv(r"C:\Users\cgray\OneDrive - Roanoke City Public Schools\10th Grade\Python\Project\Data4\unknown_differential")