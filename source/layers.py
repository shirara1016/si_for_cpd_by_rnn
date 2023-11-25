import tensorflow as tf


def truncated_interval(tTa, tTb, u):
    nominetor = -(tTa + u)
    denominetor = tTb
    plus_index = denominetor > 0
    minus_index = denominetor < 0
    u = tf.reduce_min(nominetor[plus_index] / denominetor[plus_index])
    l = tf.reduce_max(nominetor[minus_index] / denominetor[minus_index])

    return l, u


class Layer:
    def __init__(self):
        self.verbose = False

    def forward():
        pass

    def forward_si():
        pass


class Conv(Layer):
    def __init__(self, layer):
        self.strides = layer.get_config()["strides"]
        self.filters = layer.get_config()["filters"]
        self.activation = layer.get_config()["activation"]
        self.kernel_size = layer.get_config()["kernel_size"]
        self.kernel = layer.get_weights()[0]
        self.bias = layer.get_weights()[1]
        self.padding = layer.get_config()["padding"]
        self.padding = "SAME"
        self.name = layer.name

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing conv forward")

        output = tf.nn.conv2d(
            input, self.kernel, strides=self.strides, padding=self.padding
        )
        output = tf.nn.bias_add(output, self.bias)

        if self.activation == "relu":
            output = tf.nn.relu(output)
        elif self.activation == "sigmoid":
            output = tf.nn.sigmoid(output)
        elif self.activation == "softmax":
            output = tf.nn.softmax(output)

        return output

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing conv forward si")

        output_x = tf.nn.conv2d(
            x, self.kernel, strides=self.strides, padding=self.padding
        )
        output_bias = tf.nn.conv2d(
            bias, self.kernel, strides=self.strides, padding=self.padding
        )
        output_a = tf.nn.conv2d(
            a, self.kernel, strides=self.strides, padding=self.padding
        )
        output_b = tf.nn.conv2d(
            b, self.kernel, strides=self.strides, padding=self.padding
        )

        output_x = tf.nn.bias_add(output_x, self.bias)
        output_bias = tf.nn.bias_add(output_bias, self.bias)

        relu_index = output_x >= 0

        if self.activation == "relu" or self.activation == "sigmoid":
            tTa = tf.where(relu_index, -output_a, output_a)
            tTb = tf.where(relu_index, -output_b, output_b)
            bias = tf.where(relu_index, -output_bias, output_bias)

            temp_l, temp_u = truncated_interval(tTa, tTb, bias)

            l = tf.maximum(l, temp_l)
            u = tf.minimum(u, temp_u)

            if self.activation == "relu":
                output_x = tf.where(relu_index, output_x, 0)
                output_bias = tf.where(relu_index, output_bias, 0)
                output_a = tf.where(relu_index, output_a, 0)
                output_b = tf.where(relu_index, output_b, 0)
            elif self.activation == "sigmoid":
                # sigmoid をしてしまったらx以外はよくわからん値になる
                output_x = tf.nn.sigmoid(output_x)
                output_bias = tf.nn.sigmoid(output_bias)
                output_a = tf.nn.sigmoid(output_a)
                output_b = tf.nn.sigmoid(output_b)
        elif self.activation == "softmax":
            B, H, W, C = output_x.shape

            # softmaxを噛ませたらoutput_x以外もうなんの値かわからない
            max_index = tf.argmax(output_x, axis=3)
            index_range = tf.range(0, B * H * W, dtype=tf.int64)
            stack_index = tf.stack([index_range, tf.reshape(max_index, [-1])], axis=1)
            super().__init__()
            if self.verbose:
                tf.print("stack_index.shape", stack_index.shape)
            output_x_max = tf.reshape(
                tf.gather_nd(tf.reshape(output_x, [-1, C]), stack_index), [B, H, W, 1]
            )
            output_bias_max = tf.reshape(
                tf.gather_nd(tf.reshape(output_bias, [-1, C]), stack_index),
                [B, H, W, 1],
            )
            output_a_max = tf.reshape(
                tf.gather_nd(tf.reshape(output_a, [-1, C]), stack_index), [B, H, W, 1]
            )
            output_b_max = tf.reshape(
                tf.gather_nd(tf.reshape(output_b, [-1, C]), stack_index), [B, H, W, 1]
            )

            temp_l, temp_u = truncated_interval(
                output_a - output_a_max,
                output_b - output_b_max,
                output_bias - output_bias_max,
            )

            output_x = tf.nn.softmax(output_x)

            l = tf.maximum(l, temp_l)
            u = tf.minimum(u, temp_u)

        return output_x, output_bias, output_a, output_b, l, u


class MaxPool(Layer):
    def __init__(self, layer):
        self.pool_size = layer.get_config()["pool_size"]
        self.padding = layer.get_config()["padding"]
        self.strides = layer.get_config()["strides"]
        self.padding = "VALID"
        self.name = layer.name

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing MaxPool forward")
        output = tf.nn.max_pool2d(
            input, self.pool_size, strides=self.pool_size, padding=self.padding
        )

        return output

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing MaxPool forward si")

        B, H, W, C = x.shape
        H = H // self.pool_size[0]
        W = W // self.pool_size[1]

        x_im2coled = tf.image.extract_patches(
            x,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )
        bias_im2coled = tf.image.extract_patches(
            bias,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )
        a_im2coled = tf.image.extract_patches(
            a,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )
        b_im2coled = tf.image.extract_patches(
            b,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )

        x_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    x_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )
        bias_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    bias_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )
        a_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    a_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )
        b_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    b_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )

        max_index = tf.argmax(x_im2coled_reshaped, axis=1)
        add_index = tf.cast(tf.range(0, C * H * W * B), tf.int64)

        index = tf.stack([add_index, max_index], axis=1)

        output_x = tf.gather_nd(x_im2coled_reshaped, index)
        output_bias = tf.gather_nd(bias_im2coled_reshaped, index)
        output_a = tf.gather_nd(a_im2coled_reshaped, index)
        output_b = tf.gather_nd(b_im2coled_reshaped, index)

        tTa = a_im2coled_reshaped - tf.expand_dims(output_a, axis=1)
        tTb = b_im2coled_reshaped - tf.expand_dims(output_b, axis=1)
        bias = bias_im2coled_reshaped - tf.expand_dims(output_bias, axis=1)

        temp_l, temp_u = truncated_interval(tTa, tTb, bias)

        l = tf.maximum(l, temp_l)
        u = tf.minimum(u, temp_u)

        output_x = tf.reshape(output_x, [B, H, W, C])
        output_bias = tf.reshape(output_bias, [B, H, W, C])
        output_a = tf.reshape(output_a, [B, H, W, C])
        output_b = tf.reshape(output_b, [B, H, W, C])

        return output_x, output_bias, output_a, output_b, l, u


class UpSampling(Layer):
    def __init__(self, layer):
        self.size = layer.get_config()["size"]
        self.name = layer.name

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing UpSampling forward")
        output = tf.keras.layers.UpSampling2D(self.size, dtype=tf.float64)(input)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing UpSampling forward si")

        output_x = tf.keras.layers.UpSampling2D(self.size, dtype=tf.float64)(x)
        output_bias = tf.keras.layers.UpSampling2D(self.size, dtype=tf.float64)(bias)
        output_a = tf.keras.layers.UpSampling2D(self.size, dtype=tf.float64)(a)
        output_b = tf.keras.layers.UpSampling2D(self.size, dtype=tf.float64)(b)

        return output_x, output_bias, output_a, output_b, l, u


class GlobalAveragePooling2D(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing Dense")
        output = tf.reduce_mean(input, axis=[1, 2])

        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = tf.reduce_mean(x, axis=[1, 2])
        output_bias = tf.reduce_mean(bias, axis=[1, 2])
        output_a = tf.reduce_mean(a, axis=[1, 2])
        output_b = tf.reduce_mean(b, axis=[1, 2])

        return output_x, output_bias, output_a, output_b, l, u


class Flatten(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, inputs):
        B = inputs.shape[0]
        output = tf.reshape(inputs, [B, -1])

        return output

    def forward_si(self, x, bias, a, b, l, u):
        B = x.shape[0]
        output_x = tf.reshape(x, [B, -1])
        output_bias = tf.reshape(bias, [B, -1])
        output_a = tf.reshape(a, [B, -1])
        output_b = tf.reshape(b, [B, -1])

        return output_x, output_bias, output_a, output_b, l, u


class CAM(Layer):
    def __init__(self, layer, mode="thr", thr=0, k=10):
        self.name = layer.name
        self.weight = tf.cast(layer.get_weights()[0], dtype=tf.float64)
        self.mode = layer.get_config()["mode"]
        self.k = k
        self.thr = layer.get_config()["thr"]

    def forward(self, inputs):
        super().__init__()
        if self.verbose:
            print("tracing CAM")

        conv_input = inputs[0]
        output = inputs[1]

        cam_output = tf.reduce_sum(conv_input * self.weight, axis=3)

        return [cam_output, output]

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing CAM")

        conv_input = x[0]
        output = x[1]
        bias = bias[0]
        a = a[0]
        b = b[0]

        cam_output = tf.reduce_sum(conv_input * self.weight, axis=3)
        output_bias = tf.reduce_sum(bias * self.weight, axis=3)
        output_a = tf.reduce_sum(a * self.weight, axis=3)
        output_b = tf.reduce_sum(b * self.weight, axis=3)

        if self.mode == "thr":
            positive_index = cam_output >= self.thr
            tTa = tf.where(positive_index, -output_a, output_a)
            tTb = tf.where(positive_index, -output_b, output_b)
            event_bias = output_bias - self.thr
            event_bias = tf.where(positive_index, -event_bias, event_bias)

        elif self.mode == "top-k":
            # OC
            cam_reshaped = tf.reshape(cam_output, [-1])
            a_reshaped = tf.reshape(output_a, [-1])
            b_reshaped = tf.reshape(output_b, [-1])
            bias_reshaped = tf.reshape(output_bias, [-1])

            sort_index = tf.argsort(cam_reshaped, direction="DESCENDING")

            a_orderd = tf.gather(a_reshaped, sort_index)
            b_orderd = tf.gather(b_reshaped, sort_index)
            bias_orderd = tf.gather(bias_reshaped, sort_index)

            tTa = a_orderd[1:] - a_orderd[:-1]
            tTb = b_orderd[1:] - a_orderd[:-1]
            event_bias = bias_orderd[1:] - bias_orderd[:-1]

        temp_l, temp_u = truncated_interval(tTa, tTb, event_bias)

        l = tf.maximum(temp_l, l)
        u = tf.minimum(temp_u, u)

        return [cam_output, output], output_bias, output_a, output_b, l, u


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, layer):
        self.name = layer.name
        self.pool_size = (2, 2)
        self.strides = (2, 2)
        self.padding = "SAME"

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing MaxPoolingWithArgmax2D")
        output, index = tf.nn.max_pool_with_argmax(
            input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        return [output, index]

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing MaxPoolingWithArgmax2D")
        B, H, W, C = x.shape
        H = H // self.pool_size[0]
        W = W // self.pool_size[1]

        output, argmax_index = tf.nn.max_pool_with_argmax(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

        x_im2coled = tf.image.extract_patches(
            x,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )
        bias_im2coled = tf.image.extract_patches(
            bias,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )
        a_im2coled = tf.image.extract_patches(
            a,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )
        b_im2coled = tf.image.extract_patches(
            b,
            [1, self.pool_size[0], self.pool_size[1], 1],
            [1, self.strides[0], self.strides[0], 1],
            [1, 1, 1, 1],
            self.padding,
        )

        x_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    x_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )
        bias_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    bias_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )
        a_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    a_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )
        b_im2coled_reshaped = tf.reshape(
            tf.transpose(
                tf.reshape(
                    b_im2coled, [H * W * B, self.pool_size[0] * self.pool_size[1], C]
                ),
                [0, 2, 1],
            ),
            [H * W * B * C, self.pool_size[0] * self.pool_size[1]],
        )

        max_index = tf.argmax(x_im2coled_reshaped, axis=1)
        add_index = tf.cast(tf.range(0, C * H * W * B), tf.int64)

        index = tf.stack([add_index, max_index], axis=1)

        output_x = tf.gather_nd(x_im2coled_reshaped, index)
        output_bias = tf.gather_nd(bias_im2coled_reshaped, index)
        output_a = tf.gather_nd(a_im2coled_reshaped, index)
        output_b = tf.gather_nd(b_im2coled_reshaped, index)

        tTa = a_im2coled_reshaped - tf.expand_dims(output_a, axis=1)
        tTb = b_im2coled_reshaped - tf.expand_dims(output_b, axis=1)
        bias = bias_im2coled_reshaped - tf.expand_dims(output_bias, axis=1)

        temp_l, temp_u = truncated_interval(tTa, tTb, bias)

        l = tf.maximum(l, temp_l)
        u = tf.minimum(u, temp_u)

        output_x = tf.reshape(output_x, [B, H, W, C])
        output_bias = tf.reshape(output_bias, [B, H, W, C])
        output_a = tf.reshape(output_a, [B, H, W, C])
        output_b = tf.reshape(output_b, [B, H, W, C])

        return [output_x, argmax_index], output_bias, output_a, output_b, l, u


class MaxUnpooling2D(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing MaxUnpooling2D")
        input, index = input[0], input[1]
        B, H, W, C = input.shape
        super().__init__()
        if self.verbose:
            print("MaxUnpooling2D", B, H, W, C)
        input_vector = tf.reshape(input, [-1])
        index_vector = tf.reshape(index, [-1, 1])
        output = tf.scatter_nd(
            index_vector, input_vector, tf.constant([B * W * H * C * 4], dtype=tf.int64)
        )
        output = tf.reshape(output, [B, H * 2, W * 2, C])

        return output

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing MaxUnpooling2D")

        a = a[0]
        b = b[0]
        bias = bias[0]

        input_x, index = x[0], x[1]
        B, H, W, C = input_x.shape
        input_x_vector = tf.reshape(input_x, [-1])
        input_a_vector = tf.reshape(a, [-1])
        input_b_vector = tf.reshape(b, [-1])
        input_bias_vector = tf.reshape(bias, [-1])
        index_vector = tf.reshape(index, [-1, 1])
        output_x = tf.scatter_nd(
            index_vector,
            input_x_vector,
            tf.constant([B * W * H * C * 4], dtype=tf.int64),
        )
        output_x = tf.reshape(output_x, [B, H * 2, W * 2, C])
        output_a = tf.scatter_nd(
            index_vector,
            input_a_vector,
            tf.constant([B * W * H * C * 4], dtype=tf.int64),
        )
        output_a = tf.reshape(output_a, [B, H * 2, W * 2, C])
        output_b = tf.scatter_nd(
            index_vector,
            input_b_vector,
            tf.constant([B * W * H * C * 4], dtype=tf.int64),
        )
        output_b = tf.reshape(output_b, [B, H * 2, W * 2, C])
        output_bias = tf.scatter_nd(
            index_vector,
            input_bias_vector,
            tf.constant([B * W * H * C * 4], dtype=tf.int64),
        )
        output_bias = tf.reshape(output_bias, [B, H * 2, W * 2, C])

        z = tf.constant(0.14303936348024213, dtype=tf.float64)

        super().__init__()
        if self.verbose:
            tf.print(
                tf.reduce_all(
                    (output_x - (output_a + output_b * z + output_bias)) >= 1e-5
                )
            )

        return output_x, output_bias, output_a, output_b, l, u


class Dropout(Layer):
    def __init__(self, layer):
        self.rate = layer.get_config()["rate"]

    def forward(self, input):
        output = input * self.rate

        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = x * self.rate
        output_bias = bias * self.rate
        output_a = a * self.rate
        output_b = b * self.rate

        return output_x, output_bias, output_a, output_b, l, u


class BatchNorm(Layer):
    def __init__(self, layer):
        self.mean = layer.non_trinable_weights[0]
        self.variance = layer.non_trinable_weights[1]

    def forward(self, input):
        output = (input - self.mean) / self.variance

        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = (x - self.mean) / self.variance
        output_bias = (bias - self.mean) / self.variance
        output_a = a / self.variance
        output_b = b / self.variance

        return output_x, output_bias, output_a, output_b, l, u


class Dense(Layer):
    def __init__(self, layer):
        self.name = layer.name
        self.weights = tf.cast(layer.get_weights()[0], dtype=tf.float64)
        self.bias = tf.cast(layer.get_weights()[1], dtype=tf.float64)
        self.activation = layer.get_config()["activation"]

    def forward(self, input):
        super().__init__()
        if self.verbose:
            print("tracing Dense")

        output_1 = tf.add(tf.matmul(input, self.weights), self.bias)

        if self.activation == "relu":
            output_2 = tf.nn.relu(output_1)
        elif self.activation == "sigmoid":
            output_2 = tf.nn.sigmoid(output_1)
        elif self.activation == "linear":
            output_2 = output_1

        return output_2

    def forward_si(self, x, bias, a, b, l, u):
        x = tf.add(tf.matmul(x, self.weights), self.bias)
        a = tf.matmul(a, self.weights)
        b = tf.matmul(b, self.weights)
        bias = tf.add(tf.matmul(bias, self.weights), self.bias)

        active_index = x >= 0

        if self.activation == "relu" or self.activation == "sigmoid":
            tTa = tf.where(active_index, -a, a)
            tTb = tf.where(active_index, -b, b)
            bias = tf.where(active_index, -bias, bias)

            temp_l, temp_u = truncated_interval(tTa, tTb, bias)

            l = tf.maximum(l, temp_l)
            u = tf.minimum(u, temp_u)

        if self.activation == "sigmoid":
            output_x = tf.nn.sigmoid(x)
            output_a = tf.nn.sigmoid(a)
            output_b = tf.nn.sigmoid(b)
            output_bias = tf.nn.sigmoid(bias)
        elif self.activation == "relu":
            output_x = tf.where(active_index, x, 0)
            output_bias = tf.where(active_index, bias, 0)
            output_a = tf.where(active_index, a, 0)
            output_b = tf.where(active_index, b, 0)
        elif self.activation == "linear":
            output_x = x
            output_a = a
            output_b = b
            output_bias = bias

        return output_x, output_bias, output_a, output_b, l, u


class SimpleRNN(Layer):
    def __init__(self, layer):
        self.name = layer.name
        self.activation = layer.get_config()["activation"]
        self.return_sequences = layer.get_config()["return_sequences"]
        self.kernel_input = tf.cast(layer.get_weights()[0], dtype=tf.float64)
        self.kernel_hidden = tf.cast(layer.get_weights()[1], dtype=tf.float64)
        self.bias = tf.cast(layer.get_weights()[2], dtype=tf.float64)

    def forward(self, inputs):
        super().__init__()
        if self.verbose:
            print("tracing SimpleRNN")

        if self.return_sequences:
            buffer = []
            prev_h = tf.zeros(
                (inputs.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            for t in range(inputs.shape[1]):
                xt = inputs[:, t, :]
                middle_1 = tf.add(
                    tf.matmul(prev_h, self.kernel_hidden),
                    tf.matmul(xt, self.kernel_input),
                )
                middle_2 = tf.add(middle_1, self.bias)
                if self.activation == "relu":
                    ht = tf.nn.relu(middle_2)
                prev_h = ht
                ht = tf.expand_dims(ht, axis=1)
                buffer.append(ht)
            return tf.concat(buffer, axis=1)

        else:
            buffer = []
            prev_h = tf.zeros(
                (inputs.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            for t in range(inputs.shape[1]):
                xt = inputs[:, t, :]
                middle_1 = tf.add(
                    tf.matmul(prev_h, self.kernel_hidden),
                    tf.matmul(xt, self.kernel_input),
                )
                middle_2 = tf.add(middle_1, self.bias)
                if self.activation == "relu":
                    ht = tf.nn.relu(middle_2)
                prev_h = ht
            return ht

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing SimpleRNN forward si")

        if self.return_sequences:
            buffer_output = []
            buffer_a = []
            buffer_b = []
            buffer_bias = []

            prev_h = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            prev_h_bias = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            prev_h_a = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            prev_h_b = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )

            for t in range(x.shape[1]):
                xt = x[:, t, :]
                at = a[:, t, :]
                bt = b[:, t, :]
                biast = bias[:, t, :]

                output_xt = tf.add(
                    tf.matmul(prev_h, self.kernel_hidden),
                    tf.matmul(xt, self.kernel_input),
                )
                output_biast = tf.add(
                    tf.matmul(prev_h_bias, self.kernel_hidden),
                    tf.matmul(biast, self.kernel_input),
                )
                output_at = tf.add(
                    tf.matmul(prev_h_a, self.kernel_hidden),
                    tf.matmul(at, self.kernel_input),
                )
                output_bt = tf.add(
                    tf.matmul(prev_h_b, self.kernel_hidden),
                    tf.matmul(bt, self.kernel_input),
                )

                output_xt = tf.nn.bias_add(output_xt, self.bias)
                output_biast = tf.nn.bias_add(output_biast, self.bias)

                relu_index = output_xt >= 0
                if self.activation == "relu":
                    tTa = tf.where(relu_index, -output_at, output_at)
                    tTb = tf.where(relu_index, -output_bt, output_bt)
                    threshold = tf.where(relu_index, -output_biast, output_biast)

                    temp_l, temp_u = truncated_interval(tTa, tTb, threshold)

                    l = tf.maximum(l, temp_l)
                    u = tf.minimum(u, temp_u)

                    output_xt = tf.where(relu_index, output_xt, 0)
                    output_biast = tf.where(relu_index, output_biast, 0)
                    output_at = tf.where(relu_index, output_at, 0)
                    output_bt = tf.where(relu_index, output_bt, 0)

                prev_h = output_xt
                prev_h_bias = output_biast
                prev_h_a = output_at
                prev_h_b = output_bt

                buffer_output.append(tf.expand_dims(prev_h, axis=1))
                buffer_bias.append(tf.expand_dims(prev_h_bias, axis=1))
                buffer_a.append(tf.expand_dims(prev_h_a, axis=1))
                buffer_b.append(tf.expand_dims(prev_h_b, axis=1))

            output_x = tf.concat(buffer_output, axis=1)
            output_bias = tf.concat(buffer_bias, axis=1)
            output_a = tf.concat(buffer_a, axis=1)
            output_b = tf.concat(buffer_b, axis=1)

            return output_x, output_bias, output_a, output_b, l, u

        else:
            buffer_output = []
            buffer_a = []
            buffer_b = []
            buffer_bias = []

            prev_h = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            prev_h_bias = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            prev_h_a = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )
            prev_h_b = tf.zeros(
                (x.shape[0], self.kernel_hidden.shape[0]), dtype=tf.float64
            )

            for t in range(x.shape[1]):
                xt = x[:, t, :]
                at = a[:, t, :]
                bt = b[:, t, :]
                biast = bias[:, t, :]

                output_xt = tf.add(
                    tf.matmul(prev_h, self.kernel_hidden),
                    tf.matmul(xt, self.kernel_input),
                )
                output_biast = tf.add(
                    tf.matmul(prev_h_bias, self.kernel_hidden),
                    tf.matmul(biast, self.kernel_input),
                )
                output_at = tf.add(
                    tf.matmul(prev_h_a, self.kernel_hidden),
                    tf.matmul(at, self.kernel_input),
                )
                output_bt = tf.add(
                    tf.matmul(prev_h_b, self.kernel_hidden),
                    tf.matmul(bt, self.kernel_input),
                )

                output_xt = tf.nn.bias_add(output_xt, self.bias)
                output_biast = tf.nn.bias_add(output_biast, self.bias)

                relu_index = output_xt >= 0
                if self.activation == "relu":
                    tTa = tf.where(relu_index, -output_at, output_at)
                    tTb = tf.where(relu_index, -output_bt, output_bt)
                    threshold = tf.where(relu_index, -output_biast, output_biast)

                    temp_l, temp_u = truncated_interval(tTa, tTb, threshold)

                    l = tf.maximum(l, temp_l)
                    u = tf.minimum(u, temp_u)

                    output_xt = tf.where(relu_index, output_xt, 0)
                    output_biast = tf.where(relu_index, output_biast, 0)
                    output_at = tf.where(relu_index, output_at, 0)
                    output_bt = tf.where(relu_index, output_bt, 0)

                prev_h = output_xt
                prev_h_bias = output_biast
                prev_h_a = output_at
                prev_h_b = output_bt

            return output_xt, output_biast, output_at, output_bt, l, u


class ReconstructErrorMap(Layer):
    def __init__(self, layer):
        self.name = layer.name
        self.mode = layer.get_config()["mode"]
        self.thr = layer.get_config()["thr"]
        self.k = layer.get_config()["k"]

    def forward(self, inputs):
        input = inputs[0]
        predict = inputs[1]
        if self.mode == "thr":
            error = tf.abs(input - predict)
            mapping = tf.cast(error > self.thr, dtype=tf.int64)
        if self.mode == "top-k":
            pass
        return mapping

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing ReconstructErrorMap forward si")

        input = x[0]
        predict = x[1]
        output_bias = bias[0] - bias[1]
        output_a = a[0] - a[1]
        output_b = b[0] - b[1]
        error = input - predict

        if self.mode == "thr":
            mapping = tf.cast(tf.abs(error) > self.thr, dtype=tf.int64)
        elif self.mode == "top-k":
            pass

        if self.mode == "thr":
            upper_index = error >= self.thr
            tTa = tf.where(upper_index, -output_a, output_a)
            tTb = tf.where(upper_index, -output_b, output_b)
            upper_bias = output_bias - self.thr
            upper_bias = tf.where(upper_index, -upper_bias, upper_bias)

            temp_l, temp_u = truncated_interval(tTa, tTb, upper_bias)
            l = tf.maximum(temp_l, l)
            u = tf.minimum(temp_u, u)

            lower_index = error <= -self.thr
            tTa = tf.where(lower_index, output_a, -output_a)
            tTb = tf.where(lower_index, output_b, -output_b)
            lower_bias = output_bias + self.thr
            lower_bias = tf.where(lower_index, lower_bias, -lower_bias)

            temp_l, temp_u = truncated_interval(tTa, tTb, lower_bias)
            l = tf.maximum(temp_l, l)
            u = tf.minimum(temp_u, u)

        elif self.mode == "top-k":
            pass

        return mapping, output_bias, output_a, output_b, l, u


class Input(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, input):
        return input

    def forward_si(self, x, bias, a, b, l, u):
        return x, bias, a, b, l, u


class Concatenate(Layer):
    def __init__(self, layer):
        self.axis = layer.get_config()["axis"]
        self.name = layer.name

    def forward(self, inputs):
        super().__init__()
        if self.verbose:
            print("tracing Concatenate forward")
        output = tf.concat(inputs, axis=self.axis)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        super().__init__()
        if self.verbose:
            print("tracing Concatenate forward si")
        output_x = tf.concat(x, axis=self.axis)
        output_bias = tf.concat(bias, axis=self.axis)
        output_a = tf.concat(a, axis=self.axis)
        output_b = tf.concat(b, axis=self.axis)
        return output_x, output_bias, output_a, output_b, l, u


class ConvTranspose(Layer):
    def __init__(self, layer):
        self.strides = layer.get_config()["strides"]
        self.filters = layer.get_config()["filters"]
        self.activation = layer.get_config()["activation"]
        self.kernel_size = layer.get_config()["kernel_size"]
        self.kernel = tf.cast(layer.get_weights()[0], dtype=tf.float64)
        self.bias = tf.cast(layer.get_weights()[1], dtype=tf.float64)
        self.padding = layer.get_config()["padding"]
        self.padding = "SAME"
        self.name = layer.name

    # TODO`
    def forward(self, inputs):
        super().__init__()
        if self.verbose:
            print("H", type(inputs.shape[1]))
            print("strides", type(self.strides[0]))
            print("kernel_size", type(self.kernel_size[0]))
            print("padding", self.padding[0])

        B, H, W, C = inputs.shape
        new_h = (H - 1) * self.strides[0] + self.kernel_size[0]
        new_w = (W - 1) * self.strides[1] + self.kernel_size[1]
        self.output_shape = [B, new_h, new_w, self.filters]

        output = tf.nn.conv2d_transpose(
            inputs, self.kernel, self.output_shape, self.strides, padding=self.padding
        )
        output = tf.nn.bias_add(output, self.bias)

        if self.activation == "relu":
            output = tf.nn.relu(output)
        elif self.activation == "sigmoid":
            output = tf.nn.sigmoid(output)
            pass

        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = tf.nn.conv2d_transpose(
            x,
            self.kernel,
            output_shape=self.output_shape,
            strides=self.strides,
            padding=self.padding,
        )
        output_bias = tf.nn.conv2d_transpose(
            bias,
            self.kernel,
            output_shape=self.output_shape,
            strides=self.strides,
            padding=self.padding,
        )
        output_a = tf.nn.conv2d_transpose(
            a,
            self.kernel,
            output_shape=self.output_shape,
            strides=self.strides,
            padding=self.padding,
        )
        output_b = tf.nn.conv2d_transpose(
            b,
            self.kernel,
            output_shape=self.output_shape,
            strides=self.strides,
            padding=self.padding,
        )

        output_x = tf.nn.bias_add(output_x, self.bias)
        output_bias = tf.nn.bias_add(output_bias, self.bias)

        relu_index = output_x >= 0

        if self.activation is not None:
            tTa = tf.where(relu_index, -output_a, output_a)
            tTb = tf.where(relu_index, -output_b, output_b)
            bias = tf.where(relu_index, -output_bias, output_bias)

            temp_l, temp_u = truncated_interval(tTa, tTb, bias)

            l = tf.maximum(l, temp_l)
            u = tf.minimum(u, temp_u)

        if self.activation == "relu":
            output_x = tf.where(relu_index, output_x, 0)
            output_bias = tf.where(relu_index, output_bias, 0)
            output_a = tf.where(relu_index, output_a, 0)
            output_b = tf.where(relu_index, output_b, 0)
        elif self.activation == "sigmoid":
            # sigmoid をしてしまったらx以外はよくわからん値になる
            output_x = tf.nn.sigmoid(output_x)
            output_bias = tf.nn.sigmoid(output_bias)
            output_a = tf.nn.sigmoid(output_a)
            output_b = tf.nn.sigmoid(output_b)

        return output_x, output_bias, output_a, output_b, l, u


class Add(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, inputs):
        output = tf.add_n(inputs)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = tf.add_n(x)
        output_bias = tf.add_n(bias)
        output_a = tf.add_n(a)
        output_b = tf.add_n(b)

        return output_x, output_bias, output_a, output_b


class sigmoid(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, inputs):
        output = tf.math.sigmoid(inputs)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        active_index = x >= 0
        negative_index = x <= 0

        tTa = tf.where(active_index, -a, a)
        tTb = tf.where(active_index, -b, b)
        bias = tf.where(active_index, -bias, bias)

        temp_l, temp_u = truncated_interval(tTa, tTb, bias)

        l = tf.maximum(l, temp_l)
        u = tf.minimum(u, temp_u)

        output_x = tf.math.sigmoid(x)
        output_bias = tf.math.sigmoid(bias)
        output_a = tf.math.sigmoid(a)
        output_b = tf.math.sigmoid(b)

        return output_x, output_bias, output_a, output_b, l, u


class relu(Layer):
    def __init__(self, layer):
        self.name = layer.name

    def forward(self, inputs):
        output = tf.keras.layers.ReLU(inputs)

    def forward_si(self, x, bias, a, b, l, u):
        active_index = x >= 0
        negative_index = x <= 0

        tTa = tf.where(active_index, -a, a)
        tTb = tf.where(active_index, -b, b)
        bias = tf.where(active_index, -bias, bias)

        temp_l, temp_u = truncated_interval(tTa, tTb, bias)

        l = tf.maximum(l, temp_l)
        u = tf.minimum(u, temp_u)

        output_x = tf.where(active_index, x, 0)
        output_bias = tf.where(active_index, bias, 0)
        output_a = tf.where(active_index, a, 0)
        output_b = tf.where(active_index, b, 0)

        return output_x, output_bias, output_a, output_b, l, u
