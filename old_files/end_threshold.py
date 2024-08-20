class EndThreshold(tf.keras.layers.Layer):
    def __init__(self, batch_size, pos, threshold, mode="greater", reset=True, **kwargs):
        super().__init__(**kwargs)
        self.end_signal = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.batch_size = batch_size
        self.pos = pos
        self.threshold = threshold
        self.mode = mode
        self.reset = reset

    def compute_output_shape(self, in_shape):
        # Subtract 1 from last dimension of shape
        return in_shape[:-1] + [in_shape[-1] - 1]

    def ended(self):
        return self.end_signal

    def get_end_mask(self, data):
        # Gets all threads where the end unit is past threshold

        # Gets end signals
        end_signals = tf.boolean_mask(data, self.signal_mask, axis=len(data.shape) - 1)

        # Function for finding end signals
        if self.mode == "greater":
            func = tf.greater
        elif self.mode == "less":
            func = tf.less

        # Sets threshold
        thres = tf.constant(self.threshold, dtype=tf.float32)

        # Gets ended thread mask
        end_mask = func(end_signals, thres)

        # Include only active threads if active threads have been set
        try:
            end_mask = tf.logical_and(end_mask, self.active_threads)
        except AttributeError:
            pass

        return end_mask

    def get_signal_mask(self, shape):
        # Gets location of end signals based on position

        # Creates numpy mask with end signal position
        mask = np.zeros(shape, dtype=bool)
        mask[self.pos] = True

        # Converts to tensor
        tensor_mask = tf.convert_to_tensor(mask, dtype=tf.bool)
        return tensor_mask

    def build(self, in_shape):
        # Defines thread saving variables
        batch_shape = self.compute_output_shape([self.batch_size] + in_shape[1:])

        self.ended_threads = tf.Variable(tf.zeros(batch_shape, dtype=tf.float32), trainable=False)
        self.active_threads = tf.Variable(tf.ones([self.batch_size, 1], dtype=tf.bool), trainable=False)

        # Defines signal mask
        self.signal_mask = tf.Variable(self.get_signal_mask(in_shape[-1]), trainable=False)
    
    def call(self, data):
        # Makes sure end signal is false
        self.end_signal.assign(tf.constant(False, dtype=tf.bool))

        # Gets threads with end signals (returns boolean tensor mask)
        end_mask = self.get_end_mask(data)

        # Removes end signals from data
        not_signal_mask = tf.logical_not(self.signal_mask)
        data = tf.boolean_mask(data, not_signal_mask, axis=1)

        # Resets non-active threads to saved state
        data = tf.where(self.active_threads, data, self.ended_threads)
        
        # Adds ended threads to current
        self.ended_threads.assign_add(tf.where(end_mask, data, 0.))

        # Updates active threads
        not_end_mask = tf.logical_not(end_mask)

        # Uses logical and to update active threads
        self.active_threads.assign(tf.logical_and(self.active_threads, not_end_mask))

        # If no threads are active, set end signal to True
        if tf.reduce_all(tf.logical_not(self.active_threads)):
            self.end_signal.assign(tf.constant(True, dtype=tf.bool))

        return data
