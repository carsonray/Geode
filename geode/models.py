from matplotlib.pyplot import xcorr
from .core import *
from tensorflow import keras as K
from collections import OrderedDict
from .layers import MultiWrapper, MultiRecursionBlock, MultiDense
from .layers import MultiHebb2, MultiHebbQ, MultiConv2D, OneHotEncoder

class ModelRunner:
    def __call__(self, *args, **kwargs):
        model = self.get(*args, **kwargs)

        return model

    def get(self):
        pass

    def compile(self, model):
        pass

    def train(self, name, model, dataset, db_table=None, database=None, callbacks=[], **kwargs):
        handler = TrainHandler(name, db_table=db_table, database=database, clear=True)
        model.fit(dataset, callbacks=callbacks + [handler.callback], verbose=1, **kwargs)
        return handler
    
    def test(self, name, model, dataset, db_table=None, database=None, callbacks=[], **kwargs):
        handler = TestHandler(name, db_table=db_table, database=database, clear=True)
        model.evaluate(dataset, callbacks=callbacks + [handler.callback], **kwargs)
        return handler
    
    def predict(self, name, model, dataset, database=None, db_table=None, **kwargs):
        handler = ClassPredictHandler(name, dataset, database=database, db_table=db_table, clear=True)
        predictions = model.predict(handler.dataset)
        handler.add_predictions(predictions)
        return handler


class FashionDense1(ModelRunner):
    def get(self, name="fashion_dense1", **kwargs):
        model = K.Sequential([
            K.layers.Flatten(input_shape=(28, 28)),
            K.layers.Dense(128, activation='relu'),
            K.layers.Dense(20, activation='relu'),
            K.layers.Dense(10)
        ], name=name, **kwargs)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])
        
class FassionAssoc1(ModelRunner):
    def get(self, name="fassion_assoc1", **kwargs):
        # Gets image and label inputs
        img_in = K.Input(shape=[28, 28, 1], name="img_in", dtype=tf.float32)
        label_in = K.Input(shape=[10], name="label_in", dtype=tf.float32)
        type_in = K.Input(shape=[2], name="type_in", dtype=tf.float32)

        img = K.layers.Flatten(input_shape=(28, 28))(img_in)
        concat = K.layers.Concatenate(axis=-1)([img, label_in, type_in])
        x = K.layers.Dense(128, activation='relu')(concat)
        x = K.layers.Dense(20, activation='relu')(x)
        x = K.layers.Dense(10)(x)

        model = AssocModel([img_in, label_in, type_in], x, name=name, **kwargs)

        return model

    def compile(self, model):
        model.compile(optimizer='adam',
            metrics=['accuracy'])

class MultiProcess1(ModelRunner):
    def get(self, name="multi_process1"):
        model = K.Sequential([
            K.layers.Flatten(input_shape=(28, 28)),
            MultiRecursionBlock([
                MultiDense(128, activation='relu'),
                MultiDense(20, activation='relu'),
                MultiDense(10)
            ], steps=8, preserve_input=True)
        ], name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class MultiHebbian2(ModelRunner):
    def get(self, name="multi_hebbian1", batch_size=None):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32, batch_size=batch_size)
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        out = MultiRecursionBlock([
            MultiDense(128, activation="relu"),
            MultiHebb2(20, activation='relu'),
            MultiHebb2(10, activation='relu')
        ], steps=8, preserve_input=True)(x)

        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])
        
class MultiHebbianQ(ModelRunner):
    def get(self, name="multi_hebb_Q", batch_size=None):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32, batch_size=batch_size)
        label_in = K.Input(shape=[10], dtype=tf.float32, batch_size=batch_size)
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        rblock = MultiRecursionBlock([
            MultiHebbQ(128, activation="relu"),
            MultiHebbQ(20, activation='relu'),
            MultiHebbQ(10, activation='relu')
        ], steps=8, preserve_input=True)
        out = rblock(x, label=label_in)

        model = HebbianModel([img_in, label_in], out, name=name, rblock=rblock)

        return model

    def compile(self, model):
        model.compile(optimizer='adam',
            trackers=[CategoricalTracker(named=False)])

class TaskConv1(ModelRunner):
    def __init__(self, task=True):
        super().__init__()
        self.use_task = task

    def get(self, tasks, name="task_conv1"):
        # Gets image input
        img_in = K.Input(shape=[32, 32, 3], name="img_in", dtype=tf.float32)
        inputs = [img_in]

        # Applies downsampling
        x = conv_down(16, 4, 2, batchnorm=False)(img_in) # (batch, 16, 16, 16)
        x = conv_down(32, 4, 2)(x) # (batch, 8, 8, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 4, 4, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 2, 2, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 1, 1, 32)


        # Flattens
        x = K.layers.Flatten()(x) # (batch, 32)

        if self.use_task:
            # Gets task input
            task_in = K.Input(shape=[], name="task", dtype=tf.int32)
            inputs.append(task_in)

            # Concatenates with task
            task = OneHotEncoder(tasks)(task_in)
            x = K.layers.Concatenate(axis=-1)([task, x])

        # Fully connected layers
        x = K.layers.Dense(20, activation="relu")(x)
        cat_out = K.layers.Dense(10, activation="relu")(x)

        # Puts into feature dict
        output = OrderedDict([("cat_out", cat_out)])

        # Forms final model
        model = TaskModel([inputs], output, name=name)

        return model

    def compile(self, model):
        model.compile(optimizer='adam', trackers=[CategoricalTracker()])

class MultiConv1(ModelRunner):
    def __init__(self, task=True):
        super().__init__()
        self.use_task = task
        
    def get(self, tasks, name="multi_conv1"):
        # Gets image input
        img_in = K.Input(shape=[32, 32, 3], name="img_in", dtype=tf.float32)
        inputs = [img_in]

        # Applies downsampling
        x = MultiRecursionBlock([
            multiconv_down(8, [4, 4], [2, 2]), # (batch, 16, 16, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 8, 8, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 4, 4, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 2, 2, 32)
            multiconv_down(32, [4, 4], [2, 2], last=True) # (batch, 1, 1, 32)
        ], steps=10, recur=False, preserve_input=True)(img_in)
        
        # Flattens
        x = K.layers.Flatten()(x) # (batch, 32)

        if self.use_task:
            # Gets task input
            task_in = K.Input(shape=[], name="task", dtype=tf.int32)
            inputs.append(task_in)

            # Concatenates with task
            task = OneHotEncoder(tasks)(task_in)
            x = K.layers.Concatenate(axis=-1)([task, x])
        
        # Fully connected layers
        cat_out = MultiRecursionBlock([
            MultiDense(20, activation="relu"),
            MultiDense(10, activation="relu")
        ], steps=4, preserve_input=True)(x)
        
        # Puts into feature dict
        output = OrderedDict([("cat_out", cat_out)])

        # Forms final model
        model = TaskModel(inputs, output, name=name)

        return model

    def compile(self, model):
        model.compile(optimizer='adam', trackers=[CategoricalTracker()])

class MultiHebbConv2(ModelRunner):
    def __init__(self, task=True):
        super().__init__()
        self.use_task = task

    def get(self, tasks, name="multi_conv1"):
        # Gets image input
        img_in = K.Input(shape=[32, 32, 3], name="img_in", dtype=tf.float32)
        inputs = [img_in]

        # Applies downsampling
        x = MultiRecursionBlock([
            multiconv_down(8, [4, 4], [2, 2]), # (batch, 16, 16, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 8, 8, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 4, 4, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 2, 2, 32)
            multiconv_down(32, [4, 4], [2, 2], last=True) # (batch, 1, 1, 32)
        ], steps=10, recur=False, preserve_input=True)(img_in)
        
        # Flattens
        x = K.layers.Flatten()(x) # (batch, 32)

        if self.use_task:
            # Gets task input
            task_in = K.Input(shape=[], name="task", dtype=tf.int32)
            inputs.append(task_in)

            # Concatenates with task
            task = OneHotEncoder(tasks)(task_in)
            x = K.layers.Concatenate(axis=-1)([task, x])
        
        # Fully connected layers
        cat_out = MultiRecursionBlock([
            MultiHebbian2(20, activation="relu"),
            MultiHebbian2(10, activation="relu")
        ], steps=4, preserve_input=True)(x)
        
        # Puts into feature dict
        output = OrderedDict([("cat_out", cat_out)])

        # Forms final model
        model = TaskModel(inputs, output, name=name)

        return model

    def compile(self, model):
        model.compile(optimizer='adam', trackers=[CategoricalTracker()])

class TaskDense1(ModelRunner):
    def get(self, tasks, name="task_dense1"):
        # Gets categorical and numeric inputs
        num_in = K.Input(shape=[6], name="num_in", dtype=tf.float32)
        cat_in = K.Input(shape=[24], name="cat_in", dtype=tf.float32)

        # Separate dense layers for numeric and categorical input
        x = K.layers.Dense(20, activation="relu")(num_in)
        x = K.layers.Dense(10, activation="relu")(x)
        num = K.layers.Dense(10, activation="relu")(x)

        x = K.layers.Dense(30, activation="relu")(cat_in)
        x = K.layers.Dense(20, activation="relu")(x)
        cat = K.layers.Dense(10, activation="relu")(x)

        # Concatenates num and cat
        x = K.layers.Concatenate(axis=-1)([num, cat])
        
        # Final layers
        x = K.layers.Dense(20, activation="relu")(x)
        x = K.layers.Dense(10, activation="relu")(x)
        x = K.layers.Dense(4, activation="relu")(x)
        
        # Splits and puts into feature dict
        num_out, cat_out = tf.split(x, [1, 3], axis=-1)

        output = OrderedDict([
            ("num_out", num_out),
            ("cat_out", cat_out)
        ])

        # Forms final model
        model = TaskModel([num_in, cat_in], output, 
                        name=name,
                        trackers=[CategoricalTracker("cat_loss", "cat_accuracy"), NumericTracker("num_loss", "num_accuracy")])

        return model

    def compile(self, model):
        model.compile(optimizer='adam')

class TaskDense2(ModelRunner):
    def get(self, tasks, name="task_dense1"):
        # Gets categorical and numeric inputs
        num_in = K.Input(shape=[6], name="num_in", dtype=tf.float32)
        cat_in = K.Input(shape=[24], name="cat_in", dtype=tf.float32)

        # Gets task input
        task_in = K.Input(shape=[], name="task", dtype=tf.int32)
        task = OneHotEncoder(tasks)(task_in)

        # Separate dense layers for numeric and categorical input
        # Gives task to each one
        x = K.layers.Concatenate(axis=-1)([task, num_in])
        x = K.layers.Dense(20, activation="relu")(x)
        x = K.layers.Dense(10, activation="relu")(x)
        num = K.layers.Dense(10, activation="relu")(x)

        x = K.layers.Concatenate(axis=-1)([task, cat_in])
        x = K.layers.Dense(30, activation="relu")(x)
        x = K.layers.Dense(20, activation="relu")(x)
        cat = K.layers.Dense(10, activation="relu")(x)

        # Concatenates num and cat
        x = K.layers.Concatenate(axis=-1)([num, cat])
        
        # Final layers
        x = K.layers.Dense(20, activation="relu")(x)
        x = K.layers.Dense(10, activation="relu")(x)
        x = K.layers.Dense(4, activation="relu")(x)
        
        # Splits and puts into feature dict
        num_out, cat_out = tf.split(x, [1, 3], axis=-1)

        output = OrderedDict([
            ("num_out", num_out),
            ("cat_out", cat_out)
        ])

        # Forms final model
        model = TaskModel([num_in, cat_in, task_in], output,
                        name=name,
                        trackers=[CategoricalTracker("cat_loss", "cat_accuracy"), NumericTracker("num_loss", "num_accuracy")])

        return model

    def compile(self, model):
        model.compile(optimizer='adam')

class TaskMultiDense1(ModelRunner):
    def get(self, tasks, name="task_dense1"):
        # Gets categorical and numeric inputs
        num_in = K.Input(shape=[6], name="num_in", dtype=tf.float32)
        cat_in = K.Input(shape=[24], name="cat_in", dtype=tf.float32)

        # Gets task input
        task_in = K.Input(shape=[], name="task", dtype=tf.int32)
        task = OneHotEncoder(tasks)(task_in)

        # Separate dense cells for numeric and categorical input
        x = K.layers.Concatenate(axis=-1)([task, num_in])
        num = MultiRecursionBlock([
            MultiDense(20, activation='relu'),
            MultiDense(10, activation='relu'),
            MultiDense(10)
        ], steps=8, preserve_input=True)(x)

        x = K.layers.Concatenate(axis=-1)([task, cat_in])
        cat = MultiRecursionBlock([
            MultiDense(30, activation='relu'),
            MultiDense(20, activation='relu'),
            MultiDense(10)
        ], steps=8, preserve_input=True)(x)

        # Concatenates task, num and cat
        x = K.layers.Concatenate(axis=-1)([num, cat])
        
        # Final layers
        x = K.layers.Dense(20, activation="relu")(x)
        x = K.layers.Dense(4, activation="relu")(x)
        
        # Splits and puts into feature dict
        num_out, cat_out = tf.split(x, [1, 3], axis=-1)

        output = OrderedDict([
            ("num_out", num_out),
            ("cat_out", cat_out)
        ])

        # Forms final model
        model = TaskModel([num_in, cat_in, task_in], output,
                        name=name,
                        trackers=[CategoricalTracker("cat_loss", "cat_accuracy"), NumericTracker("num_loss", "num_accuracy")])

        return model

    def compile(self, model):
        model.compile(optimizer='adam')

class AllTasksReg1(ModelRunner):
    def get(self, tasks, name="all_reg1"):
        # Gets image, numerical, and catergorical inputs
        img_in = K.Input(shape=[32, 32, 3], name="img_in", dtype=tf.float32)
        num_in = K.Input(shape=[6], name="num_in", dtype=tf.float32)
        cat_in = K.Input(shape=[24], name="cat_in", dtype=tf.float32)

        # Applies downsampling
        x = conv_down(16, 4, 2, batchnorm=False)(img_in) # (batch, 16, 16, 16)
        x = conv_down(32, 4, 2)(x) # (batch, 8, 8, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 4, 4, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 2, 2, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 1, 1, 32)

        # Flattens
        img = K.layers.Flatten()(x) # (batch, 32)

        # Concatenates with task, numeric, and categorical inputs
        x = K.layers.Concatenate(axis=-1)([img, num_in, cat_in])

        # Fully connected layers
        x = K.layers.Dense(60, activation="relu")(x)
        x = K.layers.Dense(30, activation="relu")(x)
        x = K.layers.Dense(20, activation="relu")(x)
        x = K.layers.Dense(11, activation="relu")(x)

        # Splits into numeric and categorical data and adds to feature dict
        num_out, cat_out = tf.split(x, [1, 10], axis=-1)

        output = OrderedDict([
            ("num_out", num_out),
            ("cat_out", cat_out)
        ])

        # Forms final model
        model = TaskModel([num_in, cat_in], output,
                        name=name,
                        trackers=[CategoricalTracker("cat_loss", "cat_accuracy"), NumericTracker("num_loss", "num_accuracy")])

        return model

    def compile(self, model):
        model.compile(optimizer='adam')

class AllTasksReg2(ModelRunner):
    def get(self, tasks, name="all_reg1"):
        # Gets image, numerical, and catergorical inputs
        img_in = K.Input(shape=[32, 32, 3], name="img_in", dtype=tf.float32)
        num_in = K.Input(shape=[6], name="num_in", dtype=tf.float32)
        cat_in = K.Input(shape=[24], name="cat_in", dtype=tf.float32)

        # Gets task input
        task_in = K.Input(shape=[], name="task", dtype=tf.int32)

        # Applies downsampling
        x = conv_down(16, 4, 2, batchnorm=False)(img_in) # (batch, 16, 16, 16)
        x = conv_down(32, 4, 2)(x) # (batch, 8, 8, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 4, 4, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 2, 2, 32)
        x = conv_down(32, 4, 2)(x) # (batch, 1, 1, 32)

        # Flattens
        img = K.layers.Flatten()(x) # (batch, 32)

        # Concatenates with task, numeric, and categorical inputs
        task = OneHotEncoder(tasks)(task_in)
        x = K.layers.Concatenate(axis=-1)([task, img, num_in, cat_in])

        # Fully connected layers
        x = K.layers.Dense(60, activation="relu")(x)
        x = K.layers.Dense(30, activation="relu")(x)
        x = K.layers.Dense(20, activation="relu")(x)
        x = K.layers.Dense(11, activation="relu")(x)

        # Splits into numeric and categorical data and adds to feature dict
        num_out, cat_out = tf.split(x, [1, 10], axis=-1)

        output = OrderedDict([
            ("num_out", num_out),
            ("cat_out", cat_out)
        ])

        # Forms final model
        model = TaskModel([num_in, cat_in, task_in], output,
                        name=name,
                        trackers=[CategoricalTracker("cat_loss", "cat_accuracy"), NumericTracker("num_loss", "num_accuracy")])

        return model

    def compile(self, model):
        model.compile(optimizer='adam')

class AllTasksMulti1(ModelRunner):
    def get(self, tasks, name="all_multi1"):
       # Gets image, numerical, and catergorical inputs
        img_in = K.Input(shape=[32, 32, 3], name="img_in", dtype=tf.float32)
        num_in = K.Input(shape=[6], name="num_in", dtype=tf.float32)
        cat_in = K.Input(shape=[24], name="cat_in", dtype=tf.float32)

        # Gets task input
        task_in = K.Input(shape=[], name="task", dtype=tf.int32)

        # Applies downsampling
        x = MultiRecursionBlock([
            multiconv_down(8, [4, 4], [2, 2]), # (batch, 16, 16, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 8, 8, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 4, 4, 16)
            multiconv_down(16, [4, 4], [2, 2]), # (batch, 2, 2, 32)
            multiconv_down(32, [4, 4], [2, 2], last=True) # (batch, 1, 1, 32)
        ], steps=10, recur=False, preserve_input=True)(img_in)
        
        # Flattens
        img = K.layers.Flatten()(x) # (batch, 32)

        # Concatenates with task, numeric, and categorical inputs
        task = OneHotEncoder(tasks)(task_in)
        x = K.layers.Concatenate(axis=-1)([task, img, num_in, cat_in])
        
        # Fully connected layers
        x = MultiRecursionBlock([
            MultiDense(60, activation="relu"),
            MultiDense(30, activation="relu"),
            MultiDense(20, activation="relu"),
            MultiDense(11, activation="relu")
        ], steps=8, preserve_input=True)(x)
        
        # Splits into numeric and categorical data and adds to feature dict
        num_out, cat_out = tf.split(x, [1, 10], axis=-1)

        output = OrderedDict([
            ("num_out", num_out),
            ("cat_out", cat_out)
        ])

        # Forms final model
        model = TaskModel([num_in, cat_in, task_in], output,
                        name=name,
                        trackers=[CategoricalTracker("cat_loss", "cat_accuracy"), NumericTracker("num_loss", "num_accuracy")])

        return model

    def compile(self, model):
        model.compile(optimizer='adam')




def conv_down(filters, size, strides=1, batchnorm=True):
    # Creates random values for kernels
    initializer = tf.random_normal_initializer(0., 0.02)

    # Adds conv layer
    result = K.Sequential([
        K.layers.Conv2D(filters, size, strides, padding='same', kernel_initializer=initializer, use_bias=False)
    ])

    # Adds batchnorm layer if applicable
    if batchnorm:
        result.add(K.layers.BatchNormalization())

    # Adds leaky relu activation
    result.add(K.layers.LeakyReLU())

    return result

class TaskModel(tf.keras.Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, **kwargs)

    def compile(self, trackers=[], weights=None, **kwargs):
        super().compile(**kwargs)

        # Defines loss trackers and metrics
        self.trackers = trackers

        # Defines weights on each tracker's loss
        self.track_weights = tf.constant([1/len(trackers)]*len(trackers)) if weights is None else tf.constant(weights)

    def tracker_losses(self, labels, pred):
        # Calculates losses from all trackers
        losses = [tracker.get_loss(labels, pred) for tracker in self.trackers]

        return losses

    def tracker_metrics(self, losses, labels, pred):
        # Updates metrics from all trackers
        for num, tracker in enumerate(self.trackers):
            tracker.update_metrics(losses[num], labels, pred)

    def train_step(self, data):
        # Unpacks data and labels
        features, labels = data

        # Gradient descent
        with tf.GradientTape() as tape:
            # Forward pass
            pred = self(features, training=True)

            # Gets losses
            losses = self.tracker_losses(labels, pred)

            # Loss is sum of losses multiplied by weights
            loss = tf.reduce_sum(tf.stack(losses, axis=0) * self.track_weights)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics and loss trackers
        self.tracker_metrics(losses, labels, pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpacks data and labels
        features, labels = data

        # Forward pass
        pred = self(features, training=False)

        # Gets losses
        losses = self.tracker_losses(labels, pred)

        # Updates loss trackers and metrics
        self.tracker_metrics(losses, labels, pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        metrics = []

        for tracker in self.trackers:
            metrics.extend(tracker.metrics)
            
        return metrics
    
class AssocModel(tf.keras.Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        # ensure trackers exists so metrics property is safe before compile
        self.trackers = []
        self.loss_metric = None
        self.accuracy_metric = None
        self.assoc_collapse_weight = 0.75
        self.assoc_spread_weight = 0.25
        
        self.label_eye = {
            "img_in": np.zeros((10, 28, 28, 1), dtype=np.float32),
            "label_in": np.eye(10, dtype=np.float32),
            "type_in": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (10, 1)),
        }

    def compile(self, **kwargs):
        super().compile(**kwargs)

        # simple loss metric so Keras sees at least one metric
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        # simple scalar metric for the computed retrieval accuracy
        self.accuracy_metric = tf.keras.metrics.Mean(name="accuracy")
        # ensure trackers list exists
        if not hasattr(self, 'trackers'):
            self.trackers = []

    def tracker_metrics(self, losses, labels=None, pred=None):
        """Update registered trackers. Accepts a single loss or a list of losses."""
        if losses is None:
            return
        if not isinstance(losses, (list, tuple)):
            losses = [losses]

        for num, tracker in enumerate(getattr(self, 'trackers', []) or []):
            try:
                tracker.update_metrics(losses[num], labels, pred)
            except Exception:
                try:
                    tracker.update_metrics(losses[num])
                except Exception:
                    pass

    def label_tensor(self, labels):
        if isinstance(labels, dict):
            label_tensor = labels.get("cat_out")
            if label_tensor is None:
                label_tensor = next(iter(labels.values()))
            return label_tensor
        return labels

    def label_indices(self, labels):
        return tf.argmax(self.label_tensor(labels), axis=-1)

    def assoc_logits(self, pred_features, pred_labels):
        pred_features = tf.expand_dims(pred_features, axis=1)
        pred_labels = tf.expand_dims(pred_labels, axis=0)
        return tf.norm(pred_features - pred_labels, axis=-1)

    def assoc_loss(self, pred_features, pred_labels):
        # Minimize same-sample Euclidean distance while keeping both branches away from collapse.
        pair_distance = tf.reduce_mean(tf.reduce_sum(tf.square(pred_features - pred_labels), axis=-1))

        feature_norms = tf.norm(pred_features, axis=-1)
        label_norms = tf.norm(pred_labels, axis=-1)
        collapse_loss = tf.reduce_mean(tf.square(feature_norms - 1.0)) + tf.reduce_mean(tf.square(label_norms - 1.0))

        # Push label prototypes apart so classes occupy distinct regions.
        proto_embeddings = self(self.label_eye, training=True)
        proto_embeddings = tf.nn.l2_normalize(proto_embeddings, axis=-1)
        proto_sim = tf.linalg.matmul(proto_embeddings, proto_embeddings, transpose_b=True)
        proto_sim = proto_sim - tf.eye(tf.shape(proto_sim)[0], dtype=proto_sim.dtype)
        spread_loss = tf.reduce_mean(tf.square(proto_sim))

        return pair_distance + self.assoc_collapse_weight * collapse_loss + self.assoc_spread_weight * spread_loss

    def assoc_accuracy(self, pred_features, pred_labels, label_indices):
        distances = self.assoc_logits(pred_features, pred_labels)
        predicted_indices = tf.argmin(distances, axis=-1)

        return tf.reduce_mean(tf.cast(tf.equal(predicted_indices, label_indices), tf.float32))
            
    def prep_data(self, features):
        # Creates dataset with features and blank labels
        features_only = features.copy()
        features_only["label_in"] = batch_zeros(features["label_in"], features["label_in"].shape[1:])
        features_only["type_in"] = tf.tile(tf.constant([[1.0, 0.0]], dtype=tf.float32), [tf.shape(features["img_in"])[0], 1])
        
        # Creates dataset with labels and blank features
        labels_only = features.copy()
        labels_only["img_in"] = batch_zeros(features["img_in"], features["img_in"].shape[1:])
        labels_only["type_in"] = tf.tile(tf.constant([[0.0, 1.0]], dtype=tf.float32), [tf.shape(features["img_in"])[0], 1])

        return features_only, labels_only

    def train_step(self, data):
        # Unpacks data and labels
        features, labels = data
        
        features_only, labels_only = self.prep_data(features)
        label_indices = self.label_indices(labels)

        # Gradient descent
        with tf.GradientTape() as tape:
            # Forward passes through labels and features only datasets
            pred_features = self(features_only, training=True)
            pred_labels = self(labels_only, training=True)

            # Gets losses
            loss = self.assoc_loss(pred_features, pred_labels)
            accuracy = self.assoc_accuracy(pred_features, pred_labels, label_indices)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update simple loss metric and return results (report batch loss)
        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(accuracy)
        results = {"loss": loss}
        for m in self.metrics:
            results[m.name] = m.result()

        return results

    def test_step(self, data):
        # Unpacks data and labels
        features, labels = data
        
        features_only, labels_only = self.prep_data(features)
        label_indices = self.label_indices(labels)

        # Forward passes through paired feature and label branches for the loss.
        pred_features = self(features_only, training=False)
        pred_labels = self(labels_only, training=False)
        
        # Computes dot-product similarity between paired branches for validation loss.
        loss = self.assoc_loss(pred_features, pred_labels)

        # Uses prototype matching for validation accuracy.
        pred_labels = self(self.label_eye, training=False)
        accuracy = self.assoc_accuracy(pred_features, pred_labels, label_indices)

        self.loss_metric.update_state(loss)
        self.accuracy_metric.update_state(accuracy)

        results = {"loss": loss, "accuracy": accuracy}
        for m in self.metrics:
            results[m.name] = m.result()

        return results
    
    @property
    def metrics(self):
        metrics = []

        # include simple loss metric if present
        if getattr(self, "loss_metric", None) is not None:
            metrics.append(self.loss_metric)

        # include simple accuracy metric if present
        if getattr(self, "accuracy_metric", None) is not None:
            metrics.append(self.accuracy_metric)

        # include any tracker metrics
        for tracker in getattr(self, "trackers", []) or []:
            metrics.extend(getattr(tracker, "metrics", []))

        return metrics
    
class HebbianModel(tf.keras.Model):
    def __init__(self, inputs, outputs, rblock=None, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.rblock = rblock

    def compile(self, trackers=[], weights=None, **kwargs):
        super().compile(**kwargs)

        # Defines loss trackers and metrics
        self.trackers = trackers

        # Defines weights on each tracker's loss
        self.track_weights = tf.constant([1/len(trackers)]*len(trackers)) if weights is None else tf.constant(weights)

    def tracker_losses(self, labels, pred):
        # Calculates losses from all trackers
        losses = [tracker.get_loss(labels, pred) for tracker in self.trackers]

        return losses

    def tracker_metrics(self, losses, labels, pred):
        # Updates metrics from all trackers
        for num, tracker in enumerate(self.trackers):
            tracker.update_metrics(losses[num], labels, pred)

    def train_step(self, data):
        self.rblock.label_on()

        # Unpacks data and labels
        features, labels = data

        # Forward pass
        pred = self([features, labels], training=True)

        # Gets losses
        losses = self.tracker_losses(labels, pred)

        # Update metrics and loss trackers
        self.tracker_metrics(losses, labels, pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Turns off label
        self.rblock.label_off()

        # Unpacks data and labels
        features, labels = data

        # Forward pass
        pred = self([features, batch_zeros(labels, labels.shape[1:])], training=False)

        # Gets losses
        losses = self.tracker_losses(labels, pred)

        # Updates loss trackers and metrics
        self.tracker_metrics(losses, labels, pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        metrics = []

        for tracker in self.trackers:
            metrics.extend(tracker.metrics)
            
        return metrics

def multiconv_down(filters, size, strides=[1, 1], last=False):
    # Creates random values for kernels
    initializer = tf.random_normal_initializer(0., 0.02)

    # Adds conv layer
    if last:
        wrapper = MultiWrapper(
            fwd_layers=[K.layers.Conv2D(filters, size, strides, kernel_initializer=initializer, padding='same')]
        )
    else:
        wrapper = MultiWrapper(
            multi_layer=MultiConv2D(filters, size, strides, kernel_initializer=initializer, padding=['same', 'same'])
        )

    # Adds relu
    wrapper.add("out", K.layers.LeakyReLU())

    return wrapper

    
