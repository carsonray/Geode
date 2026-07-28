# Carson Ray
# Fashion association model testing with the fashion mnist dataset
print("Initializing Tensorflow...\n")

import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import geode


class AssocFashionMNIST(geode.datasets.FashionMNIST):
    def as_feature_dict(self, img, label):
        return (
            OrderedDict([
                ("img_in", img),
                ("label_in", label),
                ("type_in", tf.constant([1.0, 0.0], dtype=tf.float32)),
            ]),
            OrderedDict([("cat_out", label)])
        )

dataset = geode.datasets.CombinedTasks(
    [AssocFashionMNIST()],
    in_features=OrderedDict([
        ("img_in", [28, 28, 1]),
        ("label_in", [10])
        ,("type_in", [2])
    ]),
    label_features=OrderedDict([
        ("cat_out", [10])
    ])
)

train_data = dataset.get("train", take=1024, batch_size=32, even_batches=True)
val_data = dataset.get("validate", take=256, batch_size=32, even_batches=True)
test_data = dataset.get("test", take=512, batch_size=32, even_batches=True)


def dataset_to_numpy(ds):
    feature_batches = []
    label_batches = []

    for features, labels in ds:
        feature_batches.append(tf.nest.map_structure(lambda tensor: tensor.numpy(), features))
        label_batches.append(tf.nest.map_structure(lambda tensor: tensor.numpy(), labels))

    def concat_dict(batches):
        keys = batches[0].keys()
        return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in keys}

    return concat_dict(feature_batches), concat_dict(label_batches)


train_features, train_labels = dataset_to_numpy(train_data)
val_features, val_labels = dataset_to_numpy(val_data)
test_features, test_labels = dataset_to_numpy(test_data)

runner = geode.models.FassionAssoc1()
model = runner(name="fashion_assoc1")
runner.compile(model)
model.fit(
    train_features,
    train_labels,
    validation_data=(val_features, val_labels),
    epochs=50,
    verbose=1,
)
model.evaluate(test_features, test_labels, verbose=1)


label_inputs = {
    "img_in": np.zeros((10, 28, 28, 1), dtype=np.float32),
    "label_in": np.eye(10, dtype=np.float32),
    "type_in": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (10, 1)),
}

pred_labels = model.predict(label_inputs, verbose=0)
pred_labels = np.asarray(pred_labels, dtype=np.float32)

centered = pred_labels - pred_labels.mean(axis=0, keepdims=True)
covariance = np.matmul(centered.T, centered) / max(centered.shape[0] - 1, 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
principal_axes = eigenvectors[:, np.argsort(eigenvalues)[::-1][:2]]
coords_2d = np.matmul(centered, principal_axes)

plt.figure(figsize=(8, 8))
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True, alpha=0.3)

for index, class_name in enumerate(AssocFashionMNIST.classes):
    x_coord, y_coord = coords_2d[index]
    plt.text(x_coord, y_coord, class_name, fontsize=9)

plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=35)
plt.title("PCA of pred_label vectors from label-only inputs")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xscale("symlog", linthresh=1e-2)
plt.yscale("symlog", linthresh=1e-2)
plt.axis("equal")
plt.tight_layout()
plt.savefig("fashion_assoc_label_pca.png", dpi=150)
plt.show()
