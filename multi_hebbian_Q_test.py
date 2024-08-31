# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
print("Initializing Tensorflow...\n")
import geode
import os

dataset =  geode.datasets.FashionMNIST()
train_data = dataset.get("train", batch_size=32)
val_data = dataset.get("validate", batch_size=32)
test_data = dataset.get("test", batch_size=32)

runner = geode.models.MultiHebbianQ()
model = runner(name="multi_hebbian_Q")
runner.compile(model)

params = {
    "table_root": "fashion",
    "test_name": "multi_hebbianQ",

    "runner": runner,
    "model": model,
    "dataset": dataset,
    "train_data": train_data,
    "test_data": test_data

    #"savedir": ".\\model_checkpoints",
    #"database": "model_data"
}



# Model saving filepath
#curr_dir = os.path.dirname(__file__)
#params["savedir"] = os.path.join(curr_dir, params["savedir"])


ops = geode.operations.BasicModelOps(params)

ops.train(epochs=1, validation_data=val_data)
ops.test()