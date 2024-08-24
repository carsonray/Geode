# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
print("Initializing Tensorflow...\n")
import geode
import os

dataset_list = [
    geode.datasets.Iris(),
    geode.datasets.AutoMPG(),
    geode.datasets.Titanic()
]
batch_size = 32
dataset =  geode.datasets.CombinedTasks(dataset_list)
train_data = dataset.get("train", batch_size=batch_size)
val_data = dataset.get("validate", shuffle_tasks=True, batch_size=batch_size)
test_data = dataset.get("test", shuffle_tasks=True, batch_size=batch_size)

runner = geode.models.TaskDense2()
model = runner(dataset.tasks, name="task_dense2")
runner.compile(model)

params = {
    "table_root": "regress_tasks",
    "test_name": "dense2",

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