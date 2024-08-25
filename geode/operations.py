from .core import *

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

from .databases import *


class Operation:
    params = []
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        self.setup()

    def setup(self):
        pass

class BasicModelOps(Operation):
    def setup(self):
        # Sets up databases
        if hasattr(self, "dbconfig"):
            print("\nConnecting to database\n")
            self.engine = sqlalchemy_connect(self.dbconfig, self.database)
            print("Database: {}".format(self.database))

        # Saving handler
        if hasattr(self, "savedir"):
            self.save = SaveCheckpoint(self.model, self.savedir)
            print("Checkpoint directory: {}\n".format(self.save.dir))

        # Displays test info
        print("\nTest name: {} - {}".format(self.table_root, self.test_name))
        print("Model: {} - {}".format(type(self.runner).__name__, self.model.name))
        print("Dataset: {}\n".format(self.dataset.name))

    def summary(self):
        self.model.summary()

    def restore(self):
        print("\nRestoring model from checkpoints...\n")
        # Load model
        return self.save.restore()

    def train(self, display=True, **kwargs):
        # Training
        print("\nTraining model...\n")

        db_table = None
        database = None
        if hasattr(self, "dbconfig"):
            db_table = self.table_root + "_training"
            database = self.engine
        
        callbacks = []
        if hasattr(self, "save"):
            callbacks = [self.save.callback]

        self.train_handler = self.runner.train(self.test_name, self.model, self.train_data, db_table=db_table, database=database, callbacks=callbacks, **kwargs)
        
        if display:
            print("\nDisplaying training data...\n")
            self.train_handler.display_line("epoch")
            plt.show()

    def test(self, display=True, **kwargs):
        # Testing
        print("\nTesting model...\n")

        database = None
        db_table = None
        if hasattr(self, "dbconfig"):
            database = self.engine
            db_table = self.table_root + "_testing"
        
        self.test_handler = self.runner.test(self.test_name, self.model, self.test_data, db_table=db_table, database=database, verbose=2,**kwargs)
        if display:
            print("\nDisplaying testing data...\n")
            self.test_handler.display_bar()
            plt.show()

    def predict(self, shape, display=True, **kwargs):
        # Predictions
        print("\nGetting model predictions...\n")
        db_table = self.table_root + "_predictions"
        predict_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])

        self.predict_handler = self.runner.predict(self.test_name, predict_model, (self.dataset, self.test_data), db_table=db_table, database=self.engine, **kwargs)
        if display:
            print("\nDisplaying model predictions...\n")
            self.predict_handler.display(shape)
            plt.show()