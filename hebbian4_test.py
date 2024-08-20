# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
print("Initializing Tensorflow...\n")
import omninet as omni
import os

dataset =  omni.datasets.FashionMNIST()
batch_size = 32
train_data = dataset.get("train", batch_size=batch_size, even_batches=True)
val_data = dataset.get("validate", batch_size=batch_size, even_batches=True)
test_data = dataset.get("test", batch_size=batch_size, even_batches=True)

runner = omni.models.HebbianModel4()
model = runner(name="hebbian4", batch_size=batch_size)
runner.compile(model)

params = {
    "table_root": "fashion",
    "test_name": "hebbian4",

    "runner": runner,
    "model": model,
    "dataset": dataset,
    "train_data": train_data,
    "test_data": test_data,

    "savedir": ".\\model_checkpoints",
    "database": "model_data"
}



# Model saving filepath
curr_dir = os.path.dirname(__file__)
params["savedir"] = os.path.join(curr_dir, params["savedir"])


ops = omni.operations.BasicModelOps(params)
ops.train(epochs=20, validation_data=val_data)
ops.test()