from .core import *

class RecursionBlock(tf.keras.layers.Layer):
    def __init__(self, layers, steps=1, all_outputs=False, **kwargs):
        super().__init__(**kwargs)

        self.layers = layers

        self.steps = steps

        # Whether end states of all layers are returned
        self.all_outputs = all_outputs

    def add(self, layer):
        self.layers.append(layer)

    def pop(self):
        return self.layers.pop()

    def compute_output_shape(self, shape):
        # Finds final output shape
        for layer in self.layers:
            shape = layer.compute_output_shape(shape)

        return shape
    
    def call(self, in_data):
        data = [in_data]

        # Loops through layers and appends to data
        for step in range(self.steps):
            # Resets data array to include only last element broadcasted
            # to the input shape of the first layer
            data = [broadcast_pad(data[-1], in_data)]

            for index, layer in enumerate(self.layers):
                data.append(layer(data[index]))

        # Returns final output based on all_ouputs parameter
        if self.all_outputs:
            return data
        else:
            return data[-1]

class MultiRecursionBlock(RecursionBlock):
    def __init__(self, layers, steps, stride=1, recur=True, preserve_input=False, use_label=False, all_outputs=False, **kwargs):
        super().__init__(layers, steps, all_outputs=all_outputs, **kwargs)

        self.stride = stride
        self.recur = recur
        self.preserve_input = preserve_input
        self.use_label = use_label

    def label_on(self):
        self.use_label=True

    def label_off(self):
        self.use_label=False
        self.layers[-1].label_off()

    def compute_output_shape(self, in_shape):
        self.build(in_shape)
        return self.out_shapes[-1]

    def build(self, input_shape):
        # Finds ouput shapes of all layers
        fwd_shapes = [input_shape]
        for index, layer in enumerate(self.layers):
            shape = layer.compute_output_shape(fwd_shapes[index])
            fwd_shapes.append(shape)

        self.out_shapes = fwd_shapes[1:]
        self.fwd_shapes = fwd_shapes[:-1]

        # Cycles outputs to make backwards shapes
        self.bwd_shapes = cycle(fwd_shapes[1:], -1, recur=True)

    def run_step(self, fwd_data, bwd_data):
        # Accumulates layers
        out_data = [layer(fwd, bwd) for layer, fwd, bwd in zip(self.layers, fwd_data, bwd_data)]
        
        # Propogates forward and backward data
        new_fwd_data = cycle(out_data, self.stride, recur=self.recur)
        new_bwd_data = cycle(out_data, -self.stride, recur=self.recur)

        # If input is preserved, set first forward data to input
        if self.preserve_input:
            new_fwd_data[0] = fwd_data[0]

        # Broacasts data to input shapes
        fwd_data = broadcast_pad_list(new_fwd_data, fwd_data)
        bwd_data = broadcast_pad_list(new_bwd_data, bwd_data)

        return out_data, fwd_data, bwd_data
            
    
    def call(self, input, label=None):
        # Creates blank data

        # Gets input shape
        input_shape = input.shape
    
        # Adds blank data to input for deeper layers
        blank_fwd = [batch_zeros(input, shape[1:]) for shape in self.fwd_shapes[1:]]
        blank_bwd = [batch_zeros(input, shape[1:]) for shape in self.bwd_shapes]
        fwd_data = [input] + blank_fwd
        bwd_data = blank_bwd
        out_data = [batch_zeros(input, shape[1:]) for shape in self.out_shapes]

        if self.use_label:
            # Sets label on last layer
            self.layers[-1].set_label(label)

        # Step loop

        condition = lambda out, fwd, bwd: True

        def body(out, fwd, bwd):
            out, fwd, bwd = self.run_step(fwd, bwd)
            
            return out, fwd, bwd

        # Loop command
        out_data, fwd_data, bwd_data = tf.while_loop(condition, body, 
                                                        loop_vars=(out_data, fwd_data, bwd_data),
                                                        shape_invariants=(self.out_shapes, self.fwd_shapes, self.bwd_shapes), 
                                                        maximum_iterations=self.steps)

        # Returns final output based on all_ouputs parameter
        if self.all_outputs:
            return out_data
        else:
            # Sets static output shape
            out_data[-1].set_shape(self.compute_output_shape(input_shape))

            return out_data[-1]

class MultiWrapper(tf.keras.layers.Layer):
    def __init__(self, multi_layer=None, fwd_layers=[], bwd_layers=[], out_layers=[], **kwargs):
        super().__init__(**kwargs)
        self.fwd_layers = fwd_layers.copy()
        self.bwd_layers = bwd_layers.copy()
        self.out_layers = out_layers.copy()
        self.multi_layer = multi_layer

    def add(self, where, layer):
        # Adds layers
        if where == "fwd":
            self.fwd_layers.append(layer)
        elif where == "bwd":
            self.fwd_layers.append(layer)
        elif where == "out":
            self.out_layers.append(layer)
    
    def compute_output_shape(self, fwd_shape):
        # Shapes of forward preprocessing layers
        for layer in self.fwd_layers:
            fwd_shape = layer.compute_output_shape(fwd_shape)

        # Shape output of multiprocess layer
        if not self.multi_layer is None:
            fwd_shape = self.multi_layer.compute_output_shape(fwd_shape)

        # Shapes of output layers
        for layer in self.out_layers:
            fwd_shape = layer.compute_output_shape(fwd_shape)

        # Returns final shape
        return fwd_shape

    def call(self, f_in, b_in):
        # Forward preprocessing layers
        for layer in self.fwd_layers:
            f_in = layer(f_in)

        # Backward preprocessing layers
        for layer in self.bwd_layers:
            b_in = layer(b_in)

        # Runs multiprocess layer if it exists
        if not self.multi_layer is None:
            out = self.multi_layer(f_in, b_in)
        else:
            if len(self.bwd_layers) == 0:
                out = f_in
            else:
                out = b_in

        # Output processing layers
        for layer in self.out_layers:
            out = layer(out)

        return out

class SimpleDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_bias=True, bias_initializer=None, weight_initializer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.weight_initializer = weight_initializer
        self.activation = tf.keras.activations.get(activation)

        self.built_inputs = False

        if self.use_bias:
            # Sets bias
            self.b = self.add_weight("bias", (self.units,), trainable=True, initializer=self.bias_initializer)

    def compute_output_shape(self, shape):
        # Returns calculated output
        return shape[:-1] + self.units

    def build(self, shape):
        # Defines weights
        self.w = self.add_weight("forward", (shape[-1], self.units), 
                                    trainable=True, 
                                    initializer=self.weight_initializer)

    def call(self, in_data):
        # Uses regular dense operation
        out = tf.matmul(in_data, self.w)

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        return out

class MultiDense(SimpleDense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.built_inputs = False

    def build(self, shape):
        self.built_inputs = False

    def build_input(self, f_in, b_in):
        # Sets forward and backward weights if not built
        if not self.built_inputs:
            self.built_inputs = True

            # Forward and backward shapes
            fwd_shape, bwd_shape = f_in.shape, b_in.shape

            # Defines weights if they do not exist
            self.define_weights(fwd_shape, bwd_shape)

        return f_in, b_in

    def define_weights(self, fwd_shape, bwd_shape):
        self.fwd = self.add_weight("forward", (fwd_shape[-1], self.units), 
                                    trainable=True, 
                                    initializer=self.weight_initializer)
        self.bwd = self.add_weight("backward", (bwd_shape[-1], self.units), 
                                    trainable=True,
                                    initializer = self.weight_initializer)


    def call(self, f_in, b_in):
        # Sets up weights and formats input
        f_in, b_in = self.build_input(f_in, b_in)

        # Adds forward and backward inputs weighted by weights
        out = tf.matmul(f_in, self.fwd) + tf.matmul(b_in, self.bwd)

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        return out

class MultiHebb2(MultiDense):
    def __init__(self, units, activation=None, use_bias=True, bias_initializer=None, weight_initializer=None, **kwargs):
        super().__init__(units, activation=activation, use_bias=use_bias, bias_initializer=bias_initializer, weight_initializer=weight_initializer, **kwargs)

    def define_weights(self, fwd_shape, bwd_shape):
        # Defines actual weights
        self.fwd = self.add_weight("forward", (fwd_shape[-1], self.units), 
                                    trainable=True, 
                                    initializer=self.weight_initializer)
        self.bwd = self.add_weight("backward", (bwd_shape[-1], self.units), 
                                    trainable=True,
                                    initializer = self.weight_initializer)

        # Defines previous weights and updates them
        self.prev_fwd = self.add_weight("prev_forward", (fwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")
        self.prev_fwd.assign(self.fwd)
        self.prev_bwd = self.add_weight("prev_backward", (bwd_shape[-1], self.units), 
                                    trainable=False,
                                    initializer = "zeros")
        self.prev_bwd.assign(self.bwd)

        # Hebbian delta values
        self.delta_fwd = self.add_weight("delta_fwd", (fwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")

        self.delta_bwd = self.add_weight("delta_bwd", (bwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")

    def call(self, f_in, b_in):
        # Sets up weights and formats input
        f_in, b_in = self.build_input(f_in, b_in)

        inputs = (f_in, b_in)
        weights = (self.fwd, self.bwd)
        prev_weights = (self.prev_fwd, self.prev_bwd)
        hebb_deltas = (self.delta_fwd, self.delta_bwd)

        # Updates weights using gradient weighted by hebbian correlation
        for weight, prev_weight, delta in zip(weights, prev_weights, hebb_deltas):
            # Gets loss gradient
            gradient = weight - prev_weight

            # Updates weight with hebbian weighting
            weight.assign_add(gradient*delta)

            # Saves current weight
            prev_weight.assign(weight)

        # Calls multidense weights

        # Adds forward and backward inputs weighted by weights
        out = tf.matmul(f_in, self.fwd) + tf.matmul(b_in, self.bwd)

        # Applies hebbian math to forward and backward inputs
        out_expand = tf.expand_dims(out, axis=-2)

        for in_data, weight, delta in zip(inputs, weights, hebb_deltas):
            # Gets hebbian correlation values on weights
            # Adds dimenions to correspond axes
            in_expand = tf.expand_dims(in_data, axis=-1)
            excit = out_expand * in_expand

            # Gets hebbian inhibition based on correlation of weights to output
            inhib = out_expand * weight

            # Gets weight delta step averaged over batch
            new_delta = tf.reduce_mean(excit - inhib, axis=0)

            # Assigns delta step with tanh function
            delta.assign(tf.nn.sigmoid(new_delta))

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        return out
    
class MultiHebbQ(MultiDense):
    def __init__(self, units, activation=None, use_bias=True, bias_initializer=None, weight_initializer=None, **kwargs):
        super().__init__(units, activation=activation, use_bias=False, bias_initializer=bias_initializer, weight_initializer=weight_initializer, **kwargs)

        self.use_bias = use_bias
        if use_bias:
            self.b = self.add_weight("bias", (self.units,), trainable=False, initializer=self.bias_initializer)
        self.use_label=False

    def define_weights(self, fwd_shape, bwd_shape):
        # Defines actual weights
        self.fwd = self.add_weight("forward", (fwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer=self.weight_initializer)
        self.bwd = self.add_weight("backward", (bwd_shape[-1], self.units), 
                                    trainable=False,
                                    initializer = self.weight_initializer)
        self.label = self.add_weight("label", (32, self.units),
                                     trainable=False,
                                     initializer = self.weight_initializer)
    
    def set_label(self, label):
        # Sets label
        self.use_label = True
        self.label = label

    def label_off(self):
        self.use_label = False

    def call(self, f_in, b_in):
        # Sets up weights and formats input
        f_in, b_in = self.build_input(f_in, b_in)

        inputs = (f_in, b_in)
        weights = (self.fwd, self.bwd)

        # Calls multidense weights

        # Adds forward and backward inputs weighted by weights
        out = tf.matmul(f_in, self.fwd) + tf.matmul(b_in, self.bwd)

        # Adds bias if used
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        # Uses label for training if enabled
        if self.use_label:
            train_out = self.label
        else:
            train_out = out

        # Applies hebbian math to forward and backward inputs
        out_expand = tf.expand_dims(train_out, axis=-2)

        for in_data, weight in zip(inputs, weights):
            # Gets hebbian correlation values on weights
            # Adds dimenions to correspond axes
            in_expand = tf.expand_dims(in_data, axis=-1)
            
            # Gets weight delta step averaged over batch
            delta = tf.reduce_mean(out_expand*in_expand, axis=0)

            # Adds delta correction to weight (normalized)
            weight.assign_add(tf.tanh(delta))

        # Train bias by treating as extra weight with constant input of 1
        if self.use_bias:
            b_delta = tf.reduce_mean(-tf.tanh(train_out), axis=0)
            self.b.assign_add(b_delta) # Adds to bias (normalized)

        return train_out

class MultiConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=[(1,1), (1,1)], padding=['valid', 'valid'], data_format=None, dilation_rate=[(1,1), (1,1)], 
                groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', dir="down", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dir = dir

        # Compiles parameters
        params = []
        for i in range(2):
            params.append({
                "filters": filters,
                "kernel_size": kernel_size[i],
                "strides": strides[i],
                "padding": padding[i],
                "data_format": data_format,
                "dilation_rate": dilation_rate[i],
                "groups": groups,
                "use_bias": False,
                "kernel_initializer": kernel_initializer,
                "bias_initializer": bias_initializer
            })

        # Defines forward convolution and transposed convolution layers based on direction parameter
        if self.dir == "down":
            self.f_layer = tf.keras.layers.Conv2D(**params[0])
            self.b_layer = tf.keras.layers.Conv2DTranspose(**params[1])
        elif self.dir == "up":
            self.f_layer = tf.keras.layers.Conv2DTranspose(**params[0])
            self.b_layer = tf.keras.layers.Conv2D(**params[1])
            

        if self.use_bias:
            # Sets bias
            self.b = self.add_weight("bias", self.filters, trainable=True, initializer=bias_initializer)

    def compute_output_shape(self, fwd_shape):
        return self.f_layer.compute_output_shape(fwd_shape)

    def call(self, f_in, b_in):
        # Runs forward and backward layers
        f_in = self.f_layer(f_in)
        b_in = self.b_layer(b_in)

        # Adds results together
        out= f_in + b_in

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation if exists
        if not self.activation is None:
            out = self.activation(out)

        return out

class OneHotEncoder(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def call(self, data):
        return tf.one_hot(data, self.depth)
