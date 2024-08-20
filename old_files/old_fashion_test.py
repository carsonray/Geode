# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
print("Initializing Tensorflow...\n")

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

import os

import omninet as omni
import databases as connector




# Variables
OPS = ["load"]

MODEL_NAME = "fashion1"
SAVEDIR = ".\\model_checkpoints"
DATABASE = "model_data"

TRAIN_NAME = "basic_fashion_train1"
TEST_NAME = "basic_fashion_test1"
PREDICT_NAME = "basic_fashion_predict1"
PREDICT_TABLE = "fashion_predictions"




# Sets up databases
engine = connector.sqlalchemy_connect(DATABASE)

# Model saving filepath
curr_dir = os.path.dirname(__file__)
save_dir = os.path.join(curr_dir, SAVEDIR)

# Get dataset

print("Getting fashion MNIST Dataset...\n")

fashionData = omni.datasets.FashionMNIST()
train_data = fashionData.get("train").prefetch(tf.data.AUTOTUNE)
test_data = fashionData.get("test").prefetch(tf.data.AUTOTUNE)

# Create model runner and model
runner = omni.models.FashionDense1()
model = runner(name=MODEL_NAME)


# Saving handler
save = omni.SaveCheckpoint(model, save_dir)

if "load" in OPS:
    print("Loading model from checkpoints...\n")
    # Load model
    save.load_model(model)

if "train" in OPS:
    # Training
    print("Training model...\n")
    train_handler = runner.train(TRAIN_NAME, model, (fashionData, train_data), database=engine, epochs=10, callbacks=[save.callback])

    print("Displaying training data...")

    train_handler.display_line("epoch")

    plt.show()

if "test" in OPS:
    # Testing
    print("Testing model...\n")

    test_handler = runner.test(TEST_NAME, model, (fashionData, test_data), database=engine, verbose=2)
    test_handler.display_bar(0)
    plt.show()

if "predict" in OPS:
    # Predictions
    print("Getting model predictions...\n")

    model.add(tf.keras.layers.Softmax())

    predict_handler = runner.predict(PREDICT_NAME, model, (fashionData, test_data.take(25).cache()), database=engine, db_table=PREDICT_TABLE)

    print("Displaying model predictions...\n")
    predict_handler.display((5,5))
    plt.show()

