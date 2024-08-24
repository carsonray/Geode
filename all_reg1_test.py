# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
print("Initializing Tensorflow...\n")
import geode
import os

dataset_list = [
    geode.datasets.Iris(),
    geode.datasets.AutoMPG(),
    geode.datasets.Titanic(),
    geode.datasets.MNISTDigits(),
    geode.datasets.FashionMNIST(),
    geode.datasets.CIFAR10()
]
batch_size = 32
dataset =  geode.datasets.CombinedTasks(dataset_list)
train_data = dataset.get("train", take=108000, batch_size=batch_size)
val_data = dataset.get("validate", take=12000, shuffle_tasks=True, batch_size=batch_size)
test_data = dataset.get("test", take=20000, shuffle_tasks=True, batch_size=batch_size)

runner = geode.models.AllTasksReg1()
model = runner(dataset.tasks, name="all_reg1")
runner.compile(model)

params = {
    "table_root": "all_tasks",
    "test_name": "reg1",

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


ops = geode.operations.BasicModelOps(params)
ops.summary()
ops.train(epochs=20, validation_data=val_data)
ops.test()