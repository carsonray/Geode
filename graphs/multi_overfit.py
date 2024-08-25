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

import geode
from geode.analysis import get_tests as tests
import geode.databases as connector

# Sets up databases
engine = connector.sqlalchemy_connect("model_data")
table_root = "image_tasks2"

names = [
    tests("taskconv2t", 5, use_root=False),
    tests("multiconv1t", 5, use_root=False),
    tests("hebbconv1", 5, use_root=False),
    tests("multihebbconv2", 5, use_root=False),
    tests("taskconv1t", 5, use_root=False),
    tests("multiconv1n", 5, use_root=False),
    tests("hebbconv1n", 5, use_root=False),
    tests("multihebbconv2n", 5, use_root=False)
]

labels = [
    "taskconv2",
    "multiconv1",
    "hebbconv1",
    "multihebbconv2",
    "taskconv1",
    "multiconv1n",
    "hebbconv1n",
    "multihebbconv2n"
]

# Load model data
print("Loading model data...\n")

train_list = geode.analysis.combine_data_series(names, engine, table_root + "_training", mean=False)
test_list = geode.analysis.combine_data_series(names, engine, table_root + "_testing", mean=False)
# Creates blank dataframe
overfit = pd.DataFrame()

for train, test, label in zip(train_list, test_list, labels):
    # Creates blank dataframe for data point
    overfit_point = pd.DataFrame(columns=["name", "loss_overfit", "loss_ofit_se", "accuracy_overfit", "acc_ofit_se"])
    overfit_point["name"] = [label]
    
    last_train = train[train["epoch"] == 19.0]
    loss_overfit = np.abs(last_train["loss"].to_numpy() - test["loss"].to_numpy())
    accuracy_overfit = np.abs(last_train["accuracy"].to_numpy() - test["accuracy"].to_numpy())
    
    overfit_point["loss_overfit"] = [np.mean(loss_overfit)]
    overfit_point["accuracy_overfit"] = [np.mean(accuracy_overfit)]

    overfit_point["loss_ofit_se"] = [np.std(loss_overfit)/np.sqrt(len(loss_overfit))]
    overfit_point["acc_ofit_se"] = [np.std(loss_overfit)/np.sqrt(len(accuracy_overfit))]

    overfit = overfit.append(overfit_point, ignore_index=True)

overfit.to_csv(r"C:\Users\cgray\OneDrive - Roanoke City Public Schools\10th Grade\Python\Project\Data4\task_overfit")