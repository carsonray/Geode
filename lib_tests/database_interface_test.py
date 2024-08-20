import sqlalchemy as sql

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import omninet as omni
import databases as connector

test_engine = connector.sqlalchemy_connect("interface_test")

info = {
    "model": "dense_model",
    "dataset": "fashion_mnist"
}

print("Loading handler")
handler = omni.DataHandler("test_handler", 
                            database=test_engine, 
                            columns=["epoch", "loss", "accuracy"], 
                            info=info)

print("Adding data")
handler.add({
    "epoch": 0,
    "loss": 0.5,
    "accuracy": 0.3,
})

print("Creating new handler")
load_handler = omni.DataHandler("test_handler", database=test_engine, load=True)
print(load_handler.info)
print(load_handler.values)