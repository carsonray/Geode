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
    def __init__(self, layers, steps, stride=1, recur=True, preserve_input=False, all_outputs=False, **kwargs):
        super().__init__(layers, steps, all_outputs=all_outputs, **kwargs)

        self.stride = stride
        self.recur = recur
        self.preserve_input = preserve_input

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
            
    
    def call(self, inputs):
        # Creates blank data

        # Gets input shape
        input_shape = inputs.shape
    
        # Adds blank data to input for deeper layers
        blank_fwd = [batch_zeros(inputs, shape[1:]) for shape in self.fwd_shapes[1:]]
        blank_bwd = [batch_zeros(inputs, shape[1:]) for shape in self.bwd_shapes]
        fwd_data = [inputs] + blank_fwd
        bwd_data = blank_bwd
        out_data = [batch_zeros(inputs, shape[1:]) for shape in self.out_shapes]

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

class Hebbian(SimpleDense):
    def __init__(self, units, rate=0.01, **kwargs):
        super().__init__(units, **kwargs)
        self.rate = rate

    def build(self, shape):
        # Defines weight
        self.w = self.add_weight("weight", (shape[-1], self.units), 
                                    trainable=True, 
                                    initializer=self.weight_initializer)
        
        # Previous weight value
        self.prev_weight = self.add_weight("prev", (shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")

        # Saves current weight
        self.prev_weight.assign(self.w)

        # Hebbian delta value
        self.delta = self.add_weight("delta", (shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")

    def call(self, in_data):
        # Updates weights using hebbian delta values weighted by loss gradient

        # Gets loss gradient and applies sigmoid to reciprocal gradient
        gradient = tf.math.sigmoid(self.w - self.prev_weight)

        # Updates weight with hebbian delta and leg rate
        self.w.assign_add(self.rate * self.delta * gradient)

        # Saves current weight
        self.prev_weight.assign(self.w)

        # Uses regular dense operation
        out = tf.matmul(in_data, self.w)

        # Gets hebbian excitation values on weights
        # Adds dimenions to correspond axes
        in_expand = tf.expand_dims(in_data, axis=-1)
        out_expand = tf.expand_dims(out, axis=-2)
        excit = out_expand * in_expand

        # Gets hebbian inhibition based on correlation of weights to output
        out_correl = tf.reduce_sum(self.w * out_expand, axis=-1)
        out_correl = tf.expand_dims(out_correl, axis=-1)
        inhib = out_expand * out_correl

        # Gets weight delta step averaged over batch
        delta = tf.reduce_mean(excit - inhib, axis=0)

        # Assigns delta step
        self.delta.assign(delta)

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        return out

class MultiHebbian(MultiDense):
    def __init__(self, units, rate=0.01, activation=None, use_bias=True, bias_initializer=None, weight_initializer=None, **kwargs):
        super().__init__(units, activation=activation, use_bias=use_bias, bias_initializer=bias_initializer, weight_initializer=weight_initializer, **kwargs)
        self.rate = tf.constant(rate, dtype=tf.float32)

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

        # Hebbian values
        self.hebb_vals_fwd = self.add_weight("hebbian_vals_fwd", (2, fwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")

        self.hebb_vals_bwd = self.add_weight("hebbian_vals_bwd", (2, bwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="zeros")
        
        # Inhibitory and excitatory weights
        self.hebb_weights_fwd = self.add_weight("hebbian_weights_fwd", (2, fwd_shape[-1], self.units), 
                                    trainable=False, 
                                    initializer="ones")

        self.hebb_weights_bwd = self.add_weight("hebbian_weights_bwd", (2, bwd_shape[-1], self.units), 
                                    trainable=False,
                                    initializer="ones")

    def call(self, f_in, b_in):
        # Sets up weights and formats input
        f_in, b_in = self.build_input(f_in, b_in)

        inputs = (f_in, b_in)
        weights = (self.fwd, self.bwd)
        prev_weights = (self.prev_fwd, self.prev_bwd)
        hebb_vals = (self.hebb_vals_fwd, self.hebb_vals_bwd)
        hebb_weights = (self.hebb_weights_fwd, self.hebb_weights_bwd)

        # Updates weights using gradient weighted by hebbian correlation
        for weight, prev_weight, hebb_val, hebb_weight in zip(weights, prev_weights, hebb_vals, hebb_weights):
            # Gets loss gradient and expands
            loss_grad = weight - prev_weight
            expand_grad = tf.expand_dims(loss_grad, axis=0)

            # Updates inhibitory and excitatory weights and gets delta value
            new_weight = tf.math.divide_no_nan(expand_grad * tf.math.softmax(hebb_weight, axis=0), hebb_val)
            hebb_weight.assign(weighted_update(hebb_weight, new_weight, self.rate))

            hebbian = tf.reduce_sum(hebb_val * hebb_weight, axis=0)
            delta = tf.math.sigmoid((loss_grad - hebbian)**2)

            # Weights gradient with delta value
            weight.assign(prev_weight + loss_grad * delta)

            prev_weight.assign(weight)

        # Calls multidense weights

        # Adds forward and backward inputs weighted by weights
        out = tf.matmul(f_in, self.fwd) + tf.matmul(b_in, self.bwd)

        # Applies hebbian math to forward and backward inputs
        out_expand = tf.expand_dims(out, axis=-2)

        for in_data, weight, hebb_val in zip(inputs, weights, hebb_vals):
            # Gets hebbian correlation values on weights
            # Adds dimenions to correspond axes
            in_expand = tf.expand_dims(in_data, axis=-1)
            excit = tf.reduce_mean(out_expand * in_expand, axis=0)

            # Gets hebbian inhibition based on correlation of weights to output
            out_correl = tf.reduce_sum(weight * out_expand, axis=-1)
            out_correl = tf.expand_dims(out_correl, axis=-1)
            inhib = -tf.reduce_mean(out_expand * out_correl, axis=0)

            # Gets weight delta step averaged over batch and assigns to variable
            vals = tf.stack([excit, inhib], axis=0)
            hebb_val.assign(vals)

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        return out

class MultiHebbian2(MultiDense):
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
            weight.assign_add(gradient * delta)

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
            out_correl = tf.reduce_sum(weight * out_expand, axis=-1)
            out_correl = tf.expand_dims(out_correl, axis=-1)
            inhib = out_expand * out_correl

            # Gets weight delta step averaged over batch
            new_delta = tf.reduce_mean(excit - inhib, axis=0)

            # Assigns delta step with tanh function
            delta.assign(tf.math.tanh(new_delta))

        # Add bias if exists
        if self.use_bias:
            out = out + self.b

        # Uses activation
        if not self.activation is None:
            out = self.activation(out)

        return out

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
