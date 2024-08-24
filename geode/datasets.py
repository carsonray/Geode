from .core import *
import os
from collections import OrderedDict
from functools import partial
import math

class DataSet:
    name = "dataset"
    features = []
    in_features = OrderedDict()
    label_features = OrderedDict()

    def __init__(self, name=None):
        self.name = self.name if name is None else name
        self.info = {}
        self.data = {}
        self.examples = {}
        print("\nInitializing {}\n".format(self.name))

    def get(self, split, take=None, batch_size=None, even_batches=False):
        data = self.data[split]

        take = self.examples[split] if take is None else take

        # Adds to take amount until batches are all the same size
        if even_batches:
            take = int(math.ceil(take / batch_size) * batch_size)

        data = expand_dataset(data, self.examples[split], take)

        if not batch_size is None:
            data = data.batch(batch_size)

        return data
    
    def process(self, split, func):
        if not self.data is None:
            self.data[split] = func(self.data[split])

    def display(self, split, shape):
        size = shape[0]*shape[1]
        dataset = self.data[split].take(size)

        for num, data in enumerate(dataset):
            plt.subplot(shape[0], shape[1], num + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            plt.imshow(data[0])
            plt.xlabel(self.classes[tf.argmax(data[1], axis=0)])

class ImgDataSet(DataSet):
    classes = []
    def as_feature_dict(self, img, label):
        return (OrderedDict([("img_in", img)]), OrderedDict([("cat_out", label)]))

class RegressDataSet(DataSet):
    def as_feature_dict(self, data, labels):
        def map_features(features, data):
            # Splits data into feature dict
            num = 0
            feature_data = OrderedDict()
            for feature, shape in features.items():
                # Sets value to data slice and increments slice start
                if len(shape) > 0:
                    value = data[num:(num + shape[0])]
                    num += shape[0]
                else:
                    value = data

                feature_data[feature] = value

            return feature_data

        # Maps input and label features
        in_data = map_features(self.in_features, data)
        label_data = map_features(self.label_features, labels)

        return in_data, label_data

# Image datasets

class FashionMNIST(ImgDataSet):
    name = "fashion_mnist"

    features = ["image", "label"]
    in_features = OrderedDict([
        ("img_in", [28, 28, 1])
    ])
    label_features = OrderedDict([
        ("cat_out", [10])
    ])

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    def __init__(self, name=None):
        super().__init__(name)

    
        data, info = tfds.load("fashion_mnist", with_info=True, as_supervised=True)
        self.data["test"] = data["test"]
        self.info = info
        train_num = self.info.splits["train"].num_examples
        self.examples["test"] = self.info.splits["test"].num_examples

        # Splits validation set off from train
        splits, nums = apply_splits_dataset(data["train"], train_num, [0.9, 0.1], with_nums=True)
        self.data["train"], self.data["validate"] = splits
        self.examples["train"], self.examples["validate"] = nums

        
        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))

        
        def map_func(img, label):
            img, label = norm_image(img, label)
            label = tf.one_hot(label, len(self.classes))

            return img, label

        apply_func = lambda ds: ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE).cache()

        self.process("train", apply_func)
        self.process("train", lambda ds: ds.shuffle(self.examples["train"]))

        self.process("validate", apply_func)

        self.process("test", apply_func)

class CIFAR10(ImgDataSet):
    name = "cifar10"
    features = ["image", "label"]

    in_features = OrderedDict([
        ("img_in", [32, 32, 3])
    ])
    label_features = OrderedDict([
        ("cat_out", [10])
    ])

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, name=None):
        super().__init__(name)

    
        data, info = tfds.load("cifar10", with_info=True, as_supervised=True)
        self.data["test"] = data["test"]
        self.info = info
        train_num = self.info.splits["train"].num_examples
        self.examples["test"] = self.info.splits["test"].num_examples

        # Splits validation set off from train
        splits, nums = apply_splits_dataset(data["train"], train_num, [0.9, 0.1], with_nums=True)
        self.data["train"], self.data["validate"] = splits
        self.examples["train"], self.examples["validate"] = nums

        
        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))

        def map_func(img, label):
            img, label = norm_image(img, label)
            label = tf.one_hot(label, len(self.classes))

            return img, label

        apply_func = lambda ds: ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE).cache()

        self.process("train", apply_func)
        self.process("train", lambda ds: ds.shuffle(self.examples["train"]))

        self.process("validate", apply_func)

        self.process("test", apply_func)


class MNISTDigits(ImgDataSet):
    name = "mnist_digits"
    features = ["image", "label"]

    in_features = OrderedDict([
        ("img_in", [28, 28, 1])
    ])
    label_features = OrderedDict([
        ("cat_out", [10])
    ])

    classes = range(10)

    def __init__(self, name=None):
        super().__init__(name)

        data, info = tfds.load("mnist", with_info=True, as_supervised=True)
        self.data["test"] = data["test"]
        self.info = info
        train_num = self.info.splits["train"].num_examples
        self.examples["test"] = self.info.splits["test"].num_examples

        # Splits validation set off from train
        splits, nums = apply_splits_dataset(data["train"], train_num, [0.9, 0.1], with_nums=True)
        self.data["train"], self.data["validate"] = splits
        self.examples["train"], self.examples["validate"] = nums

        
        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))

        def map_func(img, label):
            img, label = norm_image(img, label)
            label = tf.one_hot(label, len(self.classes))

            return img, label

        apply_func = lambda ds: ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE).cache()

        self.process("train", apply_func)
        self.process("train", lambda ds: ds.shuffle(self.examples["train"]))

        self.process("validate", apply_func)

        self.process("test", apply_func)

class DeepWeeds(ImgDataSet):
    name = "deep_weeds"
    features = ["image", "label"]

    in_features = OrderedDict([
        ("img_in", [256, 256, 3])
    ])
    label_features = OrderedDict([
        ("cat_out", [9])
    ])

    classes = ["chinese apple", "latana", "parkinsonia", "parthenium", "prickly acacia", "rubber vine", 
                "siam weed", "snake weed", "negatives"]

    def __init__(self, name=None):
        super().__init__(name)

        data, info = tfds.load("deep_weeds", split="train", with_info=True, as_supervised=True)
        self.info = info
        length = self.info.splits["train"].num_examples

        def map_func(img, label):
            img, label = norm_image(img, label)
            label = tf.one_hot(label, len(self.classes))

            return img, label

        # Normalizes images and encodes labels
        data = data.map(map_func, num_parallel_calls=tf.data.AUTOTUNE).cache()

        # Splits data into train, validation, and test sets
        splits, nums = apply_splits_dataset(data, length, [0.8, 0.1, 0.1], with_nums=True)
        self.data["train"], self.data["validate"], self.data["test"] = splits
        self.examples["train"], self.examples["validate"], self.examples["test"] = nums

        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))


# Regression datasets

class Iris(RegressDataSet):
    name = "iris"
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    in_features = OrderedDict([
        ("num_in", [4])
    ])
    label_features = OrderedDict([
        ("cat_out", [3])
    ])

    label_name = 'species'
    columns = features + [label_name]
    classes = ["Iris setosa", "Iris virginica", "Iris versicolor"]

    def __init__(self, name=None):
        super().__init__(name)

        # Loads csv files

        train_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

        train_path = get_file(train_url)

        test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

        test_path = get_file(test_url)
        
        print("Train data local file: {}".format(train_path))
        print("Test data local file: {}".format(test_path))

        # Loads datasets from csv files (skips information row)
        data, train_num = dataset_from_csv(train_path,
                                            col_names=self.columns,
                                            label_name=self.label_name,
                                            skiprows=1,
                                            shuffle=True)

        # Splits validation set off from train
        splits, nums = apply_splits_dataset(data, train_num, [0.9, 0.1], with_nums=True)
        self.data["train"], self.data["validate"] = splits
        self.examples["train"], self.examples["validate"] = nums
        
        self.data["test"], self.examples["test"] = dataset_from_csv(test_path,
                                                        col_names=self.columns,
                                                        label_name=self.label_name,
                                                        skiprows=1,
                                                        shuffle=True)
        
        # Encodes labels
        def map_func(img, label):
            label = tf.one_hot(label, len(self.classes))

            return img, label

        apply_func = lambda ds: ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)

        self.process("train", apply_func)
        self.process("validate", apply_func)
        self.process("test", apply_func)

        
        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))


class AutoMPG(RegressDataSet):
    name = "auto_mpg"
    features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

    in_features = OrderedDict([
        ("num_in", [6]),
        ("cat_in", [3])
    ])
    label_features = OrderedDict([
        ("num_out", [])
    ])

    label_name = "MPG"

    columns = [label_name] + features

    def __init__(self, name=None):
        super().__init__(name)

        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

        data_path = get_file(url)
        print("Local data file: {}".format(data_path))

        # Gets dataset from csv (converts origin column to category)
        dataset, length = dataset_from_csv(data_path,
                                    col_names=self.columns,
                                    label_name=self.label_name,
                                    na_values='?',
                                    sep=' ',
                                    comment='\t',
                                    skipinitialspace=True,
                                    add_categorical=['Origin'],
                                    shuffle=True)

        # Splits data into train, validation, and test sets
        splits, nums = apply_splits_dataset(dataset, length, [0.8, 0.1, 0.1], with_nums=True)
        self.data["train"], self.data["validate"], self.data["test"] = splits
        self.examples["train"], self.examples["validate"], self.examples["test"] = nums

        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))
        


class Titanic(RegressDataSet):
    name = "titanic"
    features = ["sex", "age", "n_siblings_spouses",	"parch", "fare", "class", "deck", "embark_town", "alone"]

    in_features = OrderedDict([
        ("num_in", [4]),
        ("cat_in", [24])
    ])
    label_features = OrderedDict([
        ("num_out", [])
    ])

    label_name = "survived"
    columns = [label_name] + features

    def __init__(self, name=None):
        super().__init__(name)

        url = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"

        data_path = get_file(url)
        print("Local data file: {}".format(data_path))

        # Gets dataset from csv
        dataset, length = dataset_from_csv(data_path,
                                            label_name=self.label_name,
                                            shuffle=True)

         # Splits data into train, validation, and test sets
        splits, nums = apply_splits_dataset(dataset, length, [0.8, 0.1, 0.1], with_nums=True)
        self.data["train"], self.data["validate"], self.data["test"] = splits
        self.examples["train"], self.examples["validate"], self.examples["test"] = nums

        print("Train examples: {}".format(self.examples["train"]))
        print("Validation examples: {}".format(self.examples["validate"]))
        print("Test examples: {}".format(self.examples["test"]))

class CombinedTasks(DataSet):
    name = "combined_tasks"

    def __init__(self, datasets, name=None, in_features=None, label_features=None):
        super().__init__(name)

        # Gets feature combinations
        self.in_features = get_combined_features(datasets, 0) if in_features is None else in_features
        self.label_features = get_combined_features(datasets, 1) if label_features is None else label_features
        self.tasks = len(datasets)
        print("Tasks: {}".format(self.tasks))

        # Combines features of datasets into tasks

        # Mapping function to convert each dataset into compatible form
        def make_compat(num, dataset, in_features, label_features, data, label):
            # Maps dataset to feature dict
            add_features = dataset.as_feature_dict(data, label)

            # Maps input and output features
            in_data = map_features(add_features[0], in_features)

            # Adds task data
            in_data["task"] = tf.constant(num, dtype=tf.int32)

            label_data = map_features(add_features[1], label_features)

            return in_data, label_data

        # Maps all datasets to compatible form
        self.data = {"train": [], "validate": [], "test": []}
        self.examples = {"train": [], "validate": [], "test": []}
        for num, dataset in enumerate(datasets):
            for split in ("train", "validate", "test"):
                data = dataset.get(split)
                # Adds number of examples in dataset
                self.examples[split].append(dataset.examples[split])
                
                map_func = partial(make_compat, num, dataset, self.in_features, self.label_features)
                data = data.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)

                self.data[split].append(data)

        print("Train examples per task: {}".format(max(self.examples["train"])))
        print("Validation examples per task: {}".format(max(self.examples["validate"])))
        print("Test examples per task: {}".format(max(self.examples["test"])))

        # Expands datasets to match sizes
        for split in ("train", "validate", "test"):
            # Gets max examples
            max_num = max(self.examples[split])

            # Loops through datasets and expands to max
            for num, dataset in enumerate(self.data[split]):
                dataset = expand_dataset(dataset,
                                         self.examples[split][num],
                                         max_num)

    def get(self, split, tasks=None, take=None, batch_size=None, even_batches=False, shuffle_tasks=False):
        # Tasks used
        tasks = range(self.tasks) if tasks is None else tasks
        tasks = [self.data[split][num] for num in tasks]
        
        # Total number of examples in all tasks
        total_num = max(self.examples[split])*len(tasks)
        take = total_num if take is None else take

        # Adds to take amount until batches are all the same size
        if even_batches:
            take = int(math.ceil(take / batch_size) * batch_size)
        
        if shuffle_tasks:
            # Samples data points randomly from datasets
            data = tf.data.Dataset.sample_from_datasets(tasks)

            # Expands amount if necessary
            data = expand_dataset(data, total_num, take)

        else:
            # Expands or compresses datasets to correct size
            selected_tasks = [expand_dataset(dataset, int(total_num/len(tasks)), int(take/len(tasks)))
                              for dataset in tasks]
            # Stacks all tasks
            data = stack_datasets(selected_tasks)

        
        # Batches data
        if not batch_size is None:
            data = data.batch(batch_size)
            

        return data


def get_file(name):
    return tf.keras.utils.get_file(fname=os.path.basename(name),
                                    origin=name)

def apply_splits_dataset(dataset, length, splits, with_nums=False):
    splits = np.array(splits)
    lengths = np.round(length * splits).astype(int)

    # Loops through splits
    split_data = []

    # Tracks movement through dataset
    num = 0
    for split in lengths:
        split_data.append(dataset.skip(num).take(split).cache())
        num += split

    # Returns splits
    if with_nums:
        return split_data, lengths
    else:
        return split_data


def dataset_from_csv(data_path, col_names=None, label_name=None, add_categorical=[], shuffle=True, **kwargs):
    # Reads data from csv into dataframe
    data = pd.read_csv(data_path, 
                        names=col_names,
                        **kwargs)

    # Drops null values
    data = data.dropna()

    # Separates into features and labels
    features = data.copy()
    labels = features.pop(label_name)

    # Converts categorical columns to strings
    features = cols_to_str(features, add_categorical)

    # Gets feature dict
    data_dict = feature_dict(features)

    # Creates preprocessing model
    preprocess_model = feature_preprocess(features)

    # Runs preprocessing
    processed_features = preprocess_model(data_dict)

    # Gets tf.dataset
    dataset = tf.data.Dataset.from_tensor_slices((processed_features, labels)).cache()

    # Shuffles if necessary
    if shuffle:
        dataset = dataset.shuffle(len(data))

    # Returns dataset
    return dataset, len(data)


def pack_features(features, label):
    features = tf.stack(list(features.values()), axis=1)

    return features, label

def map_features(add_features, features):
    # Maps features to desired shapes
    # Loops through desired feature shapes
    new_features = OrderedDict()
    for feature, shape in features.items():
        # Checks for matching feature in add_features
        # If it doesn't exist, makes it blank
        add_feature = add_features.get(feature)
        if len(shape) > 0:
            if add_feature is None:
                new_features[feature] = tf.fill(shape, np.nan)
            else:
                # Broadcasts to desired shape
                add_feature = tf.cast(add_feature, tf.float32)
                new_features[feature] = broadcast_pad(add_feature, shape=shape)
        else:
            if add_feature is None:
                new_features[feature] = tf.constant(np.nan, dtype=tf.float32)
            else:
                new_features[feature] = tf.cast(add_feature, tf.float32)

    return new_features

def norm_image(image, label):
    return tf.cast(image, tf.float32) / 255, label

def cols_to_str(df, columns):
    # Converts columns in list to string
    if len(columns) == 0:
        return df
    
    # Gets dtype dictionary
    dtypes = {name: str for name in columns}

    # Applies to dataframe
    df = df.astype(dtypes)

    return df

def feature_dict(features):
    # Creates dictionary of features from dataframe
    return {name: np.array(value) for name, value in features.items()}

def feature_preprocess(features):
    # Creates feature preprocessing model
    inputs = {}

    # Gets preprocessing inputs by datatype
    for name, column in features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    # Concatenates numeric inputs
    num_inputs = {name:input for name,input in inputs.items()
                    if input.dtype == tf.float32}

    x = tf.keras.layers.Concatenate()(list(num_inputs.values()))
    norm = tf.keras.layers.Normalization()
    norm.adapt(np.array(features[num_inputs.keys()]))
    all_num_inputs = norm(x)

    # Starts input list
    preprocessed_inputs = [all_num_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
        
        # Gets one hot encoding for string data using vocabulary lookup
        lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(features[name]))
        one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        # Appends feature to inputs
        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    # Concatenates all inputs
    preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)

    # Creates model and returns
    preprocess_model = tf.keras.Model(inputs, preprocessed_inputs_cat)

    return preprocess_model

def get_combined_features(datasets, way):
    # Combines features of datasets to get maximum compatible shapes
    features = OrderedDict()

    # Loops through features of datasets
    for dataset in datasets:
        if way == 0:
            new_features = dataset.in_features
        elif way == 1:
            new_features = dataset.label_features

        for new_feature in new_features.items():
            # Unpacks key and values of new feature
            key, value = new_feature

            # Converts value to tensor
            value = tf.constant(value, dtype=tf.int32)

            # Tries to find key in current features
            curr_value = features.get(key)

            if not curr_value is None:
                # Maximizes shapes
                value = max_shape([curr_value, value])

            # Sets key and value of current features
            features[key] = value

    return features

def expand_dataset(dataset, length, size, buffer=None):
    # Expands dataset to desired size
    buffer = length if buffer is None else buffer

    # Repeats dataset to reach desired size
    if size > length:
        dataset = dataset.repeat(math.ceil(size/length))

        # Shuffles dataset with desired buffer
        dataset = dataset.shuffle(buffer)

    # Takes desired elements from dataset
    dataset = dataset.take(size).cache()

    return dataset

def stack_datasets(datasets):
    # Stacks the list of datasets into one dataset
    stacked = datasets[0]

    for dataset in datasets[1:]:
        stacked = stacked.concatenate(dataset)

    return stacked