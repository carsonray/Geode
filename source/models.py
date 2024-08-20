from matplotlib.pyplot import xcorr
from .core import *
from tensorflow import keras as K
from collections import OrderedDict
from .layers import MultiWrapper, MultiRecursionBlock, MultiDense
from .layers import Hebbian, Hebbian2, Hebbian3, Hebbian4, MultiHebbian, MultiHebbian2, NueralMemory, MultiConv2D, OneHotEncoder

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

class FashionEval1(ModelRunner):
    def get(self, name="fashion_eval1"):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32)

        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        x = K.layers.Dense(128, activation='relu')(x)
        x = K.layers.Dense(20, activation='relu')(x)
        out = K.layers.Dense(10, activation='relu')(x)

        model = K.Model(img_in, out)

        # Gets evaluator model
        evaluator = img_eval1()

        # Combines in intrinsic evaluation network
        final = IEN(model, evaluator, name=name)

        return final

    def compile(self, model):
        model.compile(m_optimizer='adam', e_optimizer='adam',
                    m_track=CategoricalTracker("m_loss", "m_accuracy", named=False, reduction="none"),
                    e_track=NumericTracker("e_loss", "e_accuracy", named=False))

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

class MultiHebbian1(ModelRunner):
    def get(self, name="multi_hebbian1", batch_size=None):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32, batch_size=batch_size)
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        out = MultiRecursionBlock([
            MultiDense(128, activation="relu"),
            MultiHebbian(20, activation='relu'),
            MultiHebbian(10, activation='relu')
        ], steps=8, preserve_input=True)(x)

        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class MultiHebbianModel2(ModelRunner):
    def get(self, name="multi_hebbian2"):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32)
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        out = MultiRecursionBlock([
            MultiDense(128, activation="relu"),
            MultiHebbian2(20, activation='relu'),
            MultiHebbian2(10, activation='relu')
        ], steps=8, preserve_input=True)(x)

        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class HebbianModel1(ModelRunner):
    def get(self, name="hebbian1"):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32)
        
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        x = Hebbian(128, rate=0.01, activation='relu')(x)
        x = Hebbian(20, rate=0.01, activation='relu')(x)
        out = Hebbian(10, rate=0.01, activation='relu')(x)
        
        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class HebbianModel2(ModelRunner):
    def get(self, name="hebbian1"):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32)
        
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        x = Hebbian2(128, activation='relu')(x)
        x = Hebbian2(20, activation='relu')(x)
        out = Hebbian2(10, activation='relu')(x)
        
        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class HebbianModel3(ModelRunner):
    def get(self, name="hebbian1"):
        img_in = K.Input(shape=[28, 28, 1], dtype=tf.float32)
        
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        x = Hebbian3(128, activation='relu')(x)
        x = Hebbian3(20, activation='relu')(x)
        out = Hebbian3(10, activation='relu')(x)
        
        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class HebbianModel4(ModelRunner):
    def get(self, name="hebbian1", batch_size=None):
        img_in = K.Input(shape=[28, 28, 1], batch_size=batch_size, dtype=tf.float32)
        
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        x = Hebbian4(128, activation='relu')(x)
        x = Hebbian4(20, activation='relu')(x)
        out = Hebbian4(10, activation='relu')(x)
        
        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

class Memory1(ModelRunner):
    def get(self, name="memory1", batch_size=None):
        img_in = K.Input(shape=[28, 28, 1], batch_size=batch_size, dtype=tf.float32)
        
        x = K.layers.Flatten(input_shape=(28, 28))(img_in)
        x = K.layers.Dense(128, activation='relu')(x)
        x = K.layers.Dense(20, activation='relu')(x)
        x, y = tf.split(x, [10, 10], axis=-1)

        y = NueralMemory(normalize=0.05)(y)

        x = K.layers.Concatenate(axis=-1)([x, y])
        out = K.layers.Dense(10, activation='relu')(x)
        
        model = K.Model(img_in, out, name=name)

        return model

    def compile(self, model, loss=K.losses.CategoricalCrossentropy(from_logits=True)):
        model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

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


class HebbConv1(ModelRunner):
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
        x = Hebbian2(20, activation="relu")(x)
        cat_out = Hebbian2(10, activation="relu")(x)

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

class MultiHebbConv1(ModelRunner):
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
            MultiHebbian(20, activation="relu"),
            MultiHebbian(10, activation="relu")
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

class IEN(tf.keras.Model):
    def __init__(self, model, evaluator, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.evaluator = evaluator

    def compile(self, m_optimizer, e_optimizer, m_track, e_track):
        super().compile()
        self.m_optimizer = tf.keras.optimizers.get(m_optimizer)
        self.e_optimizer = tf.keras.optimizers.get(e_optimizer)
        self.m_track = m_track
        self.e_track = e_track

    def call(self, data, **kwargs):
        # Gets result from model
        return self.model(data, **kwargs)
    
    def train_step(self, data):
        # Unpacks data and labels
        in_data, labels = data

        # Gradient descent
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass of model
            pred = self(in_data, training=True)

            # Gets model loss
            m_loss = self.m_track.get_loss(labels, pred)

            # Calls evaluator to get loss prediction
            eval_in = OrderedDict([
                ("data", in_data),
                ("pred", pred)
            ])

            loss_pred = self.evaluator(eval_in, training=True)

            # Gets mean of loss_pred to use in training
            m_loss_pred = tf.reduce_mean(loss_pred, axis=0)

            # Gets evaluator loss
            e_loss = self.e_track.get_loss(m_loss, loss_pred)

        # Compute gradients for evaluator
        e_trainable_vars = self.evaluator.trainable_variables
        e_gradients = tape.gradient(e_loss, e_trainable_vars)
        
        # Update evaluator weights
        self.e_optimizer.apply_gradients(zip(e_gradients, e_trainable_vars))

        # Updates evaluator metrics
        self.e_track.update_metrics(e_loss, m_loss, loss_pred)


        # Compute gradients for model based on evaluator prediction
        m_trainable_vars = self.model.trainable_variables
        m_gradients = tape.gradient(m_loss_pred, m_trainable_vars)
        
        # Update model weights
        self.m_optimizer.apply_gradients(zip(m_gradients, m_trainable_vars))
        
        # Updates model metrics
        self.m_track.update_metrics(m_loss, labels, pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpacks data and labels
        in_data, labels = data

        # Forward pass of model
        pred = self(in_data, training=False)

        # Gets model loss
        m_loss = self.m_track.get_loss(labels, pred)

        # Calls evaluator to get loss prediction
        eval_in = OrderedDict([
            ("data", in_data),
            ("pred", pred)
        ])

        loss_pred = self.evaluator(eval_in, training=False)

        # Gets evaluator loss
        e_loss = self.e_track.get_loss(m_loss, loss_pred)

        # Updates model metrics
        self.e_track.update_metrics(e_loss, m_loss, loss_pred)
        
        
        # Updates evaluator metrics
        self.m_track.update_metrics(m_loss, labels, pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        metrics = self.m_track.metrics + self.e_track.metrics
            
        return metrics

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

def img_eval1():
    # Tries to predict loss from model input and output
    img_in = K.Input(shape=[28, 28, 1], name="data", dtype=tf.float32)
    cat_in = K.Input(shape=[10], name="pred", dtype=tf.float32)

    # Processes image
    x = K.layers.Flatten(input_shape=(28, 28))(img_in)
    x = K.layers.Dense(128, activation="relu")(x)
    x = K.layers.Dense(20, activation="relu")(x)

    # Concatenates with categorical data
    x = K.layers.Concatenate(axis=-1)([cat_in, x])
    
    # Final layers
    x = K.layers.Dense(20, activation="relu")(x)
    x = K.layers.Dense(10, activation="relu")(x)
    out = K.layers.Dense(1, activation="relu")(x)
        
    model = K.Model([img_in, cat_in], out)

    return model

    
