# Author: Carson G Ray
# Language: Python
# Edition: 1-1


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from collections import OrderedDict
from tensorflow import timestamp as clock
import sqlalchemy as sql
import matplotlib.pyplot as plt
import os


# Data handlers

class SaveCheckpoint:
    def __init__(self, model, savedir, max_to_keep=3):
        # Initializes model name and corresponding checkpoint directory
        self.name = model.name
        self.dir = f"{savedir}\\{self.name}"

        # Creates checkpoint object to save model and manager object to handle multiple checkpoints
        self.ckpt = tf.train.Checkpoint(model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.dir, max_to_keep=max_to_keep)

        # Creates callback to save model
        self.callback = SaveCallback(self.manager)
    
    def restore(self):
        """Loads most recent checkpoint"""
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            return True
        else:
            print("Restore failed")
            return False

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        print(f"Saved checkpoint to {self.manager.save()}")
        self.manager.save()

class DataHandler:
    type = "data_handler"

    def __init__(self, name, database=None, db_table="data", columns=[], load=False, clear=False, logs={}, db_params={}):
        self.name = name
        self.db = database
        self.db_table = db_table

        self.columns = columns

        self.db_params = {}
        self.db_params["index"] = False
        self.db_params.update(db_params)

        self.logs = {}
        self.logs["data_name"] = self.name
        self.logs.update(logs)

        self.callback = None

        self.values = pd.DataFrame(columns = self.columns)

        # Automatic loading
        if load:
            self.load_db()
        
        # Automatic clearing
        if clear:
            self.clear_db()

    def add(self, values):
        # Converts values to dataframe
        try:
            values = pd.DataFrame(values)
        except ValueError:
            # Adds index
            values = pd.DataFrame(values, index=[0])
        
        
        # Add to current dataframe
        self.values = pd.concat([self.values, values],
                                join="outer",
                                ignore_index=True)

        # Save to database
        self.update_db(values)

    def load_db(self):
        if not self.db is None:
            # Loads values
            query = f"select * from {self.db_table} where data_name='{self.name}'"
            self.values = pd.read_sql_query(query, self.db)

            # Strips data_name
            del self.values["data_name"]

    def update_db(self, values=None, table=None):
        if not self.db is None:
            table = self.db_table if table is None else table
            print(f"Database: {self.db.url.database}; Table: {table};")

            values = self.values.copy() if values is None else values.copy()

            # Adds additional logs
            for key, value in self.logs.items():
                values[key] = [value]*len(values)
                
            values.to_sql(table, con=self.db, if_exists="append", **self.db_params)
    
    def clear_db(self, table=None):
        if not self.db is None:
            table = self.db_table if table is None else table
            query = f"delete from {table} where data_name='{self.name}'"
            with self.db.begin() as conn:
                try:
                    conn.execute(query)
                except Exception:
                    pass

    def display_line(self, key, include=None, **kwargs):
        # Includes specific columns
        include = self.values.columns if include is None else include

        # Iterate through columns
        for col_name in include:
            if col_name != key:
                plt.plot(self.values[key].to_numpy(), self.values[col_name].to_numpy(), label=col_name, **kwargs)

        # Plots legend
        plt.legend()

    def display_bar(self, key=None, include=None, **kwargs):
        # Includes specific columns
        include = list(self.values.columns) if include is None else include

        # Gets specific row of data to plot
        if key is None:
            row = self.values.iloc[0]
        else:
            column, value = key
            include.remove(column)
            row = self.values[self.values[column] == value].iloc[0]

        # Plots data
        plt.bar(include, row[include].to_numpy(), **kwargs)

class HandlerCallback(tf.keras.callbacks.Callback):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        
        # Initializes clock
        self.start_time = clock()

    def get_time(self, reset=False):
        end_time = clock().numpy()
        diff = end_time - self.start_time
        if reset:
            self.start_time = end_time
        return diff

    def reset_time(self):
        self.start_time = clock().numpy()

class TrainHandler(DataHandler):
    type = "train_handler"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.callback = TrainCallback(self)

class TrainCallback(HandlerCallback):
    def on_epoch_begin(self, epoch, logs=None):
        self.reset_time()

    def on_epoch_end(self, epoch, logs=None):
        time = self.get_time()

        print(f"Epoch {epoch + 1:05}: saving {self.handler.name} data")

        add_logs = logs.copy()
        add_logs["epoch"] = epoch
        add_logs["time"] = time
        self.handler.add(add_logs)

class TestHandler(TrainHandler):
    type = "test_handler"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = TestCallback(self)

    def add_test(self, names, values):
        logs = {name: value for name, value in zip(names, values)}

        self.add(logs)
    
    def display_bar(self, *args, **kwargs):
        super().display_bar(*args, **kwargs)


class TestCallback(HandlerCallback):
    def on_test_begin(self, logs=None):
        self.reset_time()

    def on_test_end(self, logs=None):
        time = self.get_time()

        add_logs = logs.copy()
        add_logs["time"] = time
        self.handler.add(add_logs)



class ClassPredictHandler(TrainHandler):
    type = "predict_handler"

    def __init__(self, name, dataset, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.db_params["index"] = True
        self.data_info, self.dataset = dataset
        self.callback = None

    def add_predictions(self, predictions):
        predictions = np.array(predictions, dtype=np.float64)

        df_predictions = pd.DataFrame(predictions, columns=range(predictions.shape[1]))
        self.add(df_predictions)

    def get_predictions(self, i):
        predictions = self.values.to_numpy()[i]
        return predictions, np.argmax(predictions)
    
    def display_image(self, i, img, true_label):
        predictions, predicted_label = self.get_predictions(i)

        plt.imshow(img, cmap=plt.cm.binary)

        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.data_info.classes[predicted_label],
                                100*np.max(predictions),
                                self.data_info.classes[true_label]),
                                color=color)

    def display_predictions(self, i, true_label):
        predictions, predicted_label = self.get_predictions(i)
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions, color="#777777")
        plt.ylim([0, 1])

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def display(self, shape):
        num_images = shape[0]*shape[1]
        plt.figure(figsize=(2*2*shape[0], 2*shape[1]))

        for i, (image, label) in enumerate(self.dataset.unbatch().take(num_images)):
            label = tf.argmax(label, axis=0)
            plt.subplot(shape[0], 2*shape[1], 2*i+1)
            self.display_image(i, image, label)
            plt.subplot(shape[0], 2*shape[1], 2*i+2)
            self.display_predictions(i, label)

        plt.tight_layout()


# Metrics and loss trackers
class Tracker:
    def __init__(self, named=True):
        self.metrics = []
        self.named = named

    def init_metric(self, metric):
        self.metrics.append(metric)
        return metric
            
    def get_loss(self, labels, pred):
        pass

    def update_metrics(self, loss, labels, pred):
        pass

class TaskTracker(Tracker):
    def __init__(self, tasks, named=True):
        super().__init__(named)
        self.tasks = tasks

    def init_metric(self, func):
        # Loops through tasks and assigns metric based on map function
        task_metrics = []
        for task in range(self.tasks):
            # Appends metric to current list and global metric list
            metric = func(task)
            task_metrics.append(metric)
            self.metrics.append(metric)

    def update_task_metric(self, task, vars):
        pass

class CategoricalTracker(Tracker):
    def __init__(self, loss_name="loss", accuracy_name="accuracy", named=True, reduction="auto"):
        super().__init__(named)
        self.reduction = reduction
        self.loss = tf.keras.losses.MeanSquaredError(reduction=reduction)
        self.loss_track = self.init_metric(tf.keras.metrics.Mean(name=loss_name))
        self.accuracy = self.init_metric(tf.keras.metrics.Accuracy(name=accuracy_name))

    @tf.function
    def process_labels(self, label, pred):
        if self.named:
            label = label["cat_out"]
            pred = pred["cat_out"]
        
        # Clips prediction to label

        # Filters out nan values, and if there is no data, uses zeros
        mask = no_nan(label)
        label = none_to_zero(tf.boolean_mask(label, mask))
        pred = none_to_zero(tf.boolean_mask(pred, mask))

        return label, pred

    @tf.function
    def get_loss(self, labels, pred):
        label, pred = self.process_labels(labels, pred)

        return self.loss(label, pred)

    @tf.function
    def update_metrics(self, loss, labels, pred):
        label, pred = self.process_labels(labels, pred)
        
        # Updates loss tracker
        if self.reduction == "none":
            loss = tf.reduce_mean(loss, axis=0)
            
        self.loss_track.update_state(loss)

        # Updates accuracy metric
        self.accuracy.update_state(label, pred)

class NumericTracker(Tracker):
    def __init__(self, loss_name="loss", accuracy_name="accuracy", named=True, reduction="auto"):
        super().__init__(named)
        self.reduction = reduction
        self.loss = tf.keras.losses.MeanSquaredError(reduction=reduction)
        self.loss_track = self.init_metric(tf.keras.metrics.Mean(name=loss_name))
        self.mae = self.init_metric(tf.keras.metrics.MeanAbsoluteError(name=accuracy_name))

    @tf.function
    def process_labels(self, label, pred):
        if self.named:
            label = label["num_out"]
            pred = pred["num_out"]
        
        # Clips prediction to label

        # Filters out nan values, and if there is no data, uses zeros
        mask = no_nan(label)
        label = none_to_zero(tf.boolean_mask(label, mask))
        pred = none_to_zero(tf.boolean_mask(pred, mask))

        return label, pred

    @tf.function
    def get_loss(self, labels, pred):
        label, pred = self.process_labels(labels, pred)

        return self.loss(label, pred)

    @tf.function
    def update_metrics(self, loss, labels, pred):
        label, pred = self.process_labels(labels, pred)
        
        # Updates loss tracker
        if self.reduction == "none":
            loss = tf.reduce_mean(loss, axis=0)

        self.loss_track.update_state(loss)

        # Updates accuracy metric
        self.mae.update_state(label, pred)
        



# Tensor operations
def broadcast_pad(tensor, target=None, shape=None, where="back", **kwargs):
    if shape is None:
        try:
            target_shape = heal_shape(target.shape)
        except AttributeError:
            target_shape = tf.constant([len(target)], dtype=tf.int32)
    else:
        target_shape = tf.constant(shape, dtype=tf.int32)

    try:
        tensor_shape = heal_shape(tensor.shape)
    except AttributeError:
        tensor_shape = tf.constant([len(tensor)], dtype=tf.int32)

    # If shapes have different number of dimensions
    if len(target_shape) != len(tensor_shape):
        # Gets max shape length and pads extra dimensions of shapes with ones
        len_list = [len(target_shape), len(tensor_shape)]
        max_len = max(len_list)

        target_shape = broadcast_pad(target_shape, shape=(max_len,), where="front", constant_values=1)
        tensor_shape = broadcast_pad(tensor_shape, shape=(max_len,), where="front", constant_values=1)
        tensor = tf.reshape(tensor, tensor_shape)

    shape_diff = target_shape - tensor_shape

    # Removes negative padding

    # Finds positive values
    positive = tf.greater(shape_diff, tf.constant(0))

    # Replaces negatives with zeros
    shape_diff = tf.where(positive, shape_diff, tf.zeros(shape_diff.shape, dtype=tf.int32))

    no_pad = tf.zeros(len(target_shape), dtype=tf.int32)

    # Orders padding based on direction
    if where == "front":
        order = [shape_diff, no_pad]
    elif where == "back":
        order = [no_pad, shape_diff]
    
    pad_shape = tf.stack(order, axis=1)

    return tf.pad(tensor, pad_shape, **kwargs)

def broadcast_pad_list(tensor, target, **kwargs):
    return [broadcast_pad(s_tensor, s_target, **kwargs) for s_tensor, s_target in zip(tensor, target)]

def heal_shape(shape):
    if shape[0] == None:
        return tf.constant(list([1] + shape[1:]))
    else:
        return tf.constant(list(shape))

def pad_weight(weight, shape, initializer):
    # Adjusts weight to fit new shape by adding new initialized values

    # Gets initialized values in new shape
    initializer = tf.keras.initializers.get(initializer)
    initialized = initializer(shape)

    # Creates mask to add previous values
    old_mask = tf.ones(weight.shape, dtype=tf.bool)

    # Broadcasts padding to new shape
    old_mask = broadcast_pad(old_mask, shape=shape)
    weight = broadcast_pad(weight, shape=shape)

    # Applies mask using old values when true and new values when false
    new_weight = tf.where(old_mask, weight, initialized)

    return new_weight

def batch_zeros(data, shape):
    # Creates zeros tensor using batches of data and rest of shape
    
    # Gets size of the rest of the data shape
    flat_size = tf.reduce_prod(data.shape[1:])

    # Flattens all other dimensions except for batch
    data = tf.reshape(data, [-1, flat_size])

    # Gets just batch dimension
    batches = data[:, 0]

    # Broadcasts to desired zero shape
    batches = tf.reshape(batches, [-1] + [1]*len(shape))
    return batches * tf.zeros(shape, dtype=tf.float32)

def variant_shape(tensor):
    # Returns completely variable shape of tensor
    return tf.TensorShape([None]*len(tensor.shape))

def weighted_update(val, new_val, weight):
    # Computes a weighted average of a value and a new value 
    # based on an update weight

    return val*(1 - weight) + new_val*weight

def max_shape(shapes):
    # Broadcasts shapes to each other and finds maximum compatible shape

    # Finds maximum dimenions of shapes
    max_dims = max([len(shape) for shape in shapes])

    # Broadcasts all shapes to max dimension
    shapes = [broadcast_pad(shape, shape=[max_dims], where="front", constant_values=1) 
                for shape in shapes]
            
    # Stacks shapes
    tensor_shapes = tf.stack(shapes)

    # Finds maximum shape along first axis
    return tf.reduce_max(tensor_shapes, axis=0)

def no_nan(tensor):
    # Returns a tensor mask where there are no nan values

    # Reshapes tensor to matrix
    tensor = tf.reshape(tensor, [-1, tf.reduce_prod(tensor.shape[1:])])
    is_normal = tf.logical_not(tf.math.is_nan(tensor))

    return tf.reduce_all(is_normal, axis=-1)

def none_to_zero(tensor):
    # If tensor has no shape, replace with zero constant
    if len(tensor.shape) == 0:
        return tf.constant(0, dtype=tensor.dtype)
    else:
        return tensor

def rand_bool_mask(prob, shape):
    # Creates a random bool mask weighted by true probability

    # Creates random numbers between zero and one and gets mask
    # by which ones are less than the probability
    mask = tf.less(tf.random.uniform(shape), prob)

    return mask


# Utility functions
def zip_list(*lists):
    return [list(x) for x in zip(*lists)]

def cycle(data, stride, recur=True):
    # Takes list and propogates the indices by set stride
    # Values at the end jump to the beginning if recur is true
    # If recur is false, zeros of the specified input shapes are used

    # Gets stride direction
    abs_stride = abs(stride)
    stride_dir = int(abs_stride/stride)

    # Restrict stride by tensor length
    abs_stride = abs_stride % len(data)
    stride = abs_stride*stride_dir

    # Get start and end slices of data based off of stride direction
    slices = list(np.index_exp[:-stride, -stride:])[::stride_dir]

    # Sets main block
    main = data[slices[0]]

    # Sets recur filler
    if recur:
        filler = data[slices[1]]
    else:
        filler = [tf.zeros(1)]*abs_stride

    # Concatenates main and filler based on stride direction
    return filler + main if stride_dir == 1 else main + filler