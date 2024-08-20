# Carson Ray
# 11/16/21
# Nueral network testing with fashion mnist dataset
print("Initializing Tensorflow...\n")
import omninet as omni
import os

dataset_list = [
    omni.datasets.MNISTDigits(),
    omni.datasets.FashionMNIST(),
    omni.datasets.CIFAR10()
]

dataset =  omni.datasets.CombinedTasks(dataset_list)
train_data = dataset.get("train", take=54000, batch_size=32)
val_data = dataset.get("validate", take=6000, shuffle_tasks=True, batch_size=32)
test_data = dataset.get("test", take=12000, shuffle_tasks=True, batch_size=32)

# Model saving filepath
savedir = ".\\model_checkpoints"
curr_dir = os.path.dirname(__file__)
savedir = os.path.join(curr_dir, savedir)

params = {
    "table_root": "image_tasks2",
    
    "dataset": dataset,
    "train_data": train_data,
    "test_data": test_data,

    "savedir": savedir,
    "database": "model_data"
}

# Tests
test_starts = [0]*4
test_nums = [5]*4
runners = [
    omni.models.MultiConv1(task=False),
    omni.models.MultiHebbConv2(task=True),
    omni.models.MultiHebbConv2(task=False)
]
model_roots = [
    "multiconv1n",
    "multihebbconv2",
    "multihebbconv2n"
]
test_roots = [
    "multiconv1n",
    "multihebbconv2",
    "multihebbconv2n"
]

for test_start, test_num, runner, model_root, test_root in zip(test_starts, test_nums, runners, model_roots, test_roots):
    params.update({
        "runner": runner
    })

    # Unit testing loop
    for test in range(test_start, test_start + test_num):
        model = runner(dataset.tasks, name="{}-{}".format(model_root, test))
        runner.compile(model)
        params["model"] = model
        params["test_name"] = "{}-{}".format(test_root, test)

        ops = omni.operations.BasicModelOps(params)


        ops.train(epochs=20, validation_data=val_data, display=False)
        ops.test(display=False)

