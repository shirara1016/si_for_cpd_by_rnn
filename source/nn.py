import tensorflow as tf

from source.layers import *


class NN:
    def __init__(self, model):
        self.input = model.input_names[0]
        self.output = model.output_names[0]
        self.layers = []
        self.layers_dict = {}

        for layer in model.layers:
            type: str = layer.__class__.__name__
            name: str = layer.name
            if type == "Conv2D":
                self.layers.append(Conv(layer))
                self.layers_dict[name] = Conv(layer)
            elif type == "MaxPooling2D":
                self.layers.append(MaxPool(layer))
                self.layers_dict[name] = MaxPool(layer)
            elif type == "UpSampling2D":
                self.layers.append(UpSampling(layer))
                self.layers_dict[name] = UpSampling(layer)
            elif type == "Dense":
                self.layers.append(Dense(layer))
                self.layers_dict[name] = Dense(layer)
            elif type == "BatchNormalization":
                self.layers.append(BatchNorm(layer))
                self.layers_dict[name] = BatchNorm(layer)
            elif type == "Dropout":
                self.layers.append(Dropout(layer))
                self.layers_dict[name] = MaxPool(layer)
            elif type == "InputLayer":
                self.layers.append(Input(layer))
                self.layers_dict[name] = Input(layer)
            elif type == "Concatenate":
                self.layers.append(Concatenate(layer))
                self.layers_dict[name] = Input(layer)
            elif type == "Add":
                self.layers.append(Add(layer))
                self.layers_dict[name] = Add(layer)
            elif type == "TFOpLambda":
                self.layers.append(sigmoid(layer))
            elif type == "Conv2DTranspose":
                self.layers.append(ConvTranspose(layer))
                self.layers_dict[name] = ConvTranspose(layer)
            elif type == "MaxPoolingWithArgmax2D":
                self.layers.append(MaxPoolingWithArgmax2D(layer))
                self.layers_dict[name] = MaxPoolingWithArgmax2D(layer)
            elif type == "MaxUnpooling2D":
                self.layers.append(MaxUnpooling2D(layer))
                self.layers_dict[name] = MaxUnpooling2D(layer)
            elif type == "CAM":
                self.layers.append(CAM(layer))
                self.layers_dict[name] = CAM(layer)
            elif type == "GlobalAveragePooling2D":
                self.layers.append(GlobalAveragePooling2D(layer))
                self.layers_dict[name] = GlobalAveragePooling2D(layer)
            elif type == "Flatten":
                self.layers.append(Flatten(layer))
                self.layers_dict[name] = Flatten(layer)
            elif type == "SimpleRNN":
                self.layers.append(SimpleRNN(layer))
                self.layers_dict[name] = SimpleRNN(layer)
            elif type == "ReconstructErrorMap":
                self.layers.append(ReconstructErrorMap(layer))
                self.layers_dict[name] = ReconstructErrorMap(layer)
            else:
                assert False, "Unsupported layer type: {}".format(type)

        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

        def get_layer_summary_with_connections(layer):
            info = {}
            connections = []
            output_index = []
            for node in layer._inbound_nodes:
                if relevant_nodes and node not in relevant_nodes:
                    continue
                for (
                    inbound_layer,
                    node_index,
                    tensor_index,
                    _,
                ) in node.iterate_inbound():
                    connections.append(inbound_layer.name)
                    output_index.append(tensor_index)

            name = layer.name
            info["type"] = layer.__class__.__name__
            info["parents"] = connections
            info["output_index"] = output_index
            return info

        self.connections = {}
        layers = model.layers
        for layer in layers:
            info = get_layer_summary_with_connections(layer)
            self.connections[layer.name] = info

    @tf.function
    def forward(self, input):
        output_dict = {}
        for layer in self.layers:
            connections = self.connections[layer.name]["parents"]
            index = self.connections[layer.name]["output_index"]

            if len(connections) == 0:
                input_tensors = input
            elif len(connections) == 1:
                input_tensors = output_dict[connections[0]]
                if type(input_tensors) == type([]):
                    if len(index) == 1:
                        input_tensors = input_tensors[index[0]]
                    else:
                        input_tensors = [input_tensors[index[i]] for i in index]
            else:
                input_tensors = [output_dict[i][j] for i, j in zip(connections, index)]

            output_dict[layer.name] = layer.forward(input_tensors)

            if type(output_dict[layer.name]) != type([]):
                output_dict[layer.name] = [output_dict[layer.name]]

        output = output_dict[self.output][0]

        return output

    @tf.function
    def forward_si(self, input_si):
        output_si_dict = {}
        for layer in self.layers:
            connections = self.connections[layer.name]["parents"]
            index = self.connections[layer.name]["output_index"]

            if len(connections) == 0:
                x, bias, a, b, l, u = input_si
            elif len(connections) == 1:
                x, bias, a, b, l, u = output_si_dict[connections[0]]
                x = x[0]
            else:
                if len(index) > 1:
                    x = [output_si_dict[i][0][j] for i, j in zip(connections, index)]
                else:
                    x = [output_si_dict[i][0] for i in connections]
                bias = [output_si_dict[i][1] for i in connections]
                a = [output_si_dict[i][2] for i in connections]
                b = [output_si_dict[i][3] for i in connections]
                l_list = [output_si_dict[i][4] for i in connections]
                u_list = [output_si_dict[i][5] for i in connections]
                l = tf.reduce_max(l_list)
                u = tf.reduce_min(u_list)

            x, bias, a, b, l, u = layer.forward_si(x, bias, a, b, l, u)

            if type(x) != type([]):
                x = [x]

            output_si_dict[layer.name] = x, bias, a, b, l, u

        output_x, output_bias, output_a, output_b, l, u = output_si_dict[self.output]

        return l, u, output_x[0], output_si_dict


class ChangesBase:
    def __init__(
        self, model, look_back=10, predict_length=10, num_cp=2, smooth_window=5
    ):
        self.predictor = NN(model)
        self.look_back = look_back
        self.predict_length = predict_length
        self.num_cp = num_cp
        self.smooth_window = smooth_window

    def set_long_term_predict(self, look_back, predict_length):
        self.look_back = look_back
        self.predict_length = predict_length

    def _long_term_predict_error(self, a, b, z):
        x = a + b * z
        bias = tf.zeros(x.shape, dtype=tf.float64)
        l = tf.constant(-1000000, dtype=tf.float64)
        u = tf.constant(100000, dtype=tf.float64)

        inputs_x, references_x = [], []
        inputs_a, references_a = [], []
        inputs_b, references_b = [], []
        inputs_bias, references_bias = [], []
        for i in range(len(x) - self.look_back - self.predict_length):
            inputs_x.append(x[i : i + self.look_back])
            inputs_a.append(a[i : i + self.look_back])
            inputs_b.append(b[i : i + self.look_back])
            inputs_bias.append(bias[i : i + self.look_back])
            references_x.append(
                x[i + self.look_back : i + self.look_back + self.predict_length]
            )
            references_a.append(
                a[i + self.look_back : i + self.look_back + self.predict_length]
            )
            references_b.append(
                b[i + self.look_back : i + self.look_back + self.predict_length]
            )
            references_bias.append(
                bias[i + self.look_back : i + self.look_back + self.predict_length]
            )
        inputs_x = tf.stack(inputs_x)
        inputs_a = tf.stack(inputs_a)
        inputs_b = tf.stack(inputs_b)
        inputs_bias = tf.stack(inputs_bias)
        references_x = tf.stack(references_x)
        references_a = tf.stack(references_a)
        references_b = tf.stack(references_b)
        references_bias = tf.stack(references_bias)

        for _ in range(self.predict_length):
            _, _, _, output_dict = self.predictor.forward_si(
                (inputs_x, inputs_bias, inputs_a, inputs_b, l, u)
            )
            next_step_x, next_step_bias, next_step_a, next_step_b, l, u = output_dict[
                self.predictor.output
            ]
            next_step_x = tf.expand_dims(next_step_x[0], axis=1)
            next_step_a = tf.expand_dims(next_step_a, axis=1)
            next_step_b = tf.expand_dims(next_step_b, axis=1)
            next_step_bias = tf.expand_dims(next_step_bias, axis=1)
            inputs_x = tf.strided_slice(
                tf.concat([inputs_x, next_step_x], axis=1),
                [0, 1, 0],
                [len(x), self.look_back + 1, x.shape[1]],
            )
            inputs_a = tf.strided_slice(
                tf.concat([inputs_a, next_step_a], axis=1),
                [0, 1, 0],
                [len(x), self.look_back + 1, x.shape[1]],
            )
            inputs_b = tf.strided_slice(
                tf.concat([inputs_b, next_step_b], axis=1),
                [0, 1, 0],
                [len(x), self.look_back + 1, x.shape[1]],
            )
            inputs_bias = tf.strided_slice(
                tf.concat([inputs_bias, next_step_bias], axis=1),
                [0, 1, 0],
                [len(x), self.look_back + 1, x.shape[1]],
            )
        before_zeros = tf.zeros(
            (self.look_back, self.predict_length, x.shape[1]), dtype=tf.float64
        )
        after_zeros = tf.zeros(
            (self.predict_length, self.predict_length, x.shape[1]), dtype=tf.float64
        )
        errors = tf.concat([before_zeros, inputs_x - references_x, after_zeros], axis=0)
        errors_a = tf.concat(
            [before_zeros, inputs_a - references_a, after_zeros], axis=0
        )
        errors_b = tf.concat(
            [before_zeros, inputs_b - references_b, after_zeros], axis=0
        )
        errors_bias = tf.concat(
            [before_zeros, inputs_bias - references_bias, after_zeros], axis=0
        )
        return errors, errors_bias, errors_a, errors_b, l, u

    def _calculate_error_score(self, data):
        inputs, references = [], []
        for i in range(len(data) - self.look_back - self.predict_length):
            inputs.append(data[i : i + self.look_back])
            references.append(
                data[i + self.look_back : i + self.look_back + self.predict_length]
            )
        inputs = tf.stack(inputs)
        references = tf.stack(references)

        for _ in range(self.predict_length):
            next_step = tf.expand_dims(self.predictor.forward(inputs), axis=1)
            inputs = tf.strided_slice(
                tf.concat([inputs, next_step], axis=1),
                [0, 1, 0],
                [len(data), self.look_back + 1, data.shape[1]],
            )
        errors = tf.concat(
            [
                tf.zeros(
                    (self.look_back, self.predict_length, data.shape[1]),
                    dtype=tf.float64,
                ),
                inputs - references,
                tf.zeros(
                    (self.predict_length, self.predict_length, data.shape[1]),
                    dtype=tf.float64,
                ),
            ],
            axis=0,
        )
        error_score = tf.reshape(tf.reduce_mean(tf.square(errors), axis=1), [-1])
        return error_score

    def _smoothing_error_score(self, error_score, smoothing_window=5):
        conv_error_score = tf.nn.conv1d(
            tf.reshape(error_score, [1, -1, 1]),
            tf.ones([smoothing_window, 1, 1], dtype=tf.float64) / smoothing_window,
            1,
            "SAME",
        )
        conv_error_score = tf.reshape(conv_error_score, [-1])
        smoothed_error_score = tf.concat(
            [
                tf.zeros(self.look_back, dtype=tf.float64),
                tf.strided_slice(
                    conv_error_score,
                    [self.look_back],
                    [len(conv_error_score) - self.predict_length],
                ),
                tf.zeros(self.predict_length, dtype=tf.float64),
            ],
            axis=0,
        )
        return smoothed_error_score

    def _detect_change_points(self, data):
        data = tf.reshape(data, [-1, 1])
        error_score = self._calculate_error_score(data)
        error_score = self._smoothing_error_score(error_score, self.smooth_window)
        self.error_score = error_score

        signs = (error_score[1:] - error_score[:-1]) >= 0
        change_points = []
        for i in range(len(signs) - 1):
            if signs[i] == True and signs[i + 1] == False:
                change_points.append(i + 1)

        if len(change_points) < self.num_cp:
            return None

        maximum_error = tf.gather(error_score, change_points)
        cp_indexes = tf.argsort(maximum_error, direction="DESCENDING")[: self.num_cp]

        cps = [-1, len(data) - 1]
        for cp_index in cp_indexes:
            cps.append(change_points[cp_index])
        cps.sort()

        for i in range(1, len(cps)):
            if cps[i] - cps[i - 1] == 1:
                return None

        return cps

    def _parameterize_error_score(self, a, b, z):
        errors, errors_bias, errors_a, errors_b, l, u = self._long_term_predict_error(
            a, b, z
        )

        error_score = tf.reduce_mean(tf.square(errors), axis=1)
        errors_a = tf.add(errors_a, errors_bias)
        alpha = tf.reduce_mean(tf.square(errors_b), axis=1)
        beta = 2 * tf.reduce_mean(tf.multiply(errors_a, errors_b), axis=1)
        gamma = tf.reduce_mean(tf.square(errors_a), axis=1)

        error_score = tf.reshape(error_score, [-1])
        alpha = tf.reshape(alpha, [-1])
        beta = tf.reshape(beta, [-1])
        gamma = tf.reshape(gamma, [-1])

        return error_score, alpha, beta, gamma, z, l, u

    def _quadratics_to_interval(self, alpha, beta, gamma, z, l, u):
        alpha_zero_index = alpha == 0
        alpha_plus_index = alpha > 0
        alpha_minus_index = alpha < 0

        disc = tf.where(
            alpha_zero_index, 0.0, tf.square(beta) - 4 * tf.multiply(alpha, gamma)
        )

        beta_plus_index = beta > 0
        beta_minus_index = beta < 0

        linear_lower_index = tf.logical_and(alpha_zero_index, beta_minus_index)
        linear_upper_index = tf.logical_and(alpha_zero_index, beta_plus_index)

        temp_l1 = tf.reduce_max(-gamma[linear_lower_index] / beta[linear_lower_index])
        temp_u1 = tf.reduce_min(-gamma[linear_upper_index] / beta[linear_upper_index])

        temp_l2 = tf.reduce_max(
            (-beta[alpha_plus_index] - tf.sqrt(disc[alpha_plus_index]))
            / (2 * alpha[alpha_plus_index])
        )
        temp_u2 = tf.reduce_min(
            (-beta[alpha_plus_index] + tf.sqrt(disc[alpha_plus_index]))
            / (2 * alpha[alpha_plus_index])
        )

        convex_intersect_index = tf.logical_and(alpha_minus_index, disc > 0)

        right_intersects = (
            -beta[convex_intersect_index] - tf.sqrt(disc[convex_intersect_index])
        ) / (2 * alpha[convex_intersect_index])
        left_intersects = (
            -beta[convex_intersect_index] + tf.sqrt(disc[convex_intersect_index])
        ) / (2 * alpha[convex_intersect_index])

        temp_l3 = tf.reduce_max(right_intersects[right_intersects < z])
        temp_u3 = tf.reduce_min(left_intersects[left_intersects > z])

        l = tf.reduce_max([l, temp_l1, temp_l2, temp_l3])
        u = tf.reduce_min([u, temp_u1, temp_u2, temp_u3])
        return l, u

    def algorithm(self, a, b, z):
        a = tf.reshape(a, [-1, 1])
        b = tf.reshape(b, [-1, 1])

        error_score, alpha, beta, gamma, z, l, u = self._parameterize_error_score(
            a, b, z
        )
        error_score = self._smoothing_error_score(error_score, self.smooth_window)
        alpha = self._smoothing_error_score(alpha, self.smooth_window)
        beta = self._smoothing_error_score(beta, self.smooth_window)
        gamma = self._smoothing_error_score(gamma, self.smooth_window)

        # para_error = alpha * z ** 2 + beta * z + gamma
        # print(tf.reduce_all(tf.abs(para_error - error_score) < 1e-5))

        signs = (error_score[1:] - error_score[:-1]) >= 0

        signs_tensor = tf.where(signs, -1, 1)
        signs_tensor = tf.cast(signs_tensor, tf.float64)

        strided_alpha = (alpha[1:] - alpha[:-1]) * signs_tensor
        strided_beta = (beta[1:] - beta[:-1]) * signs_tensor
        strided_gamma = (gamma[1:] - gamma[:-1]) * signs_tensor

        l, u = self._quadratics_to_interval(
            strided_alpha, strided_beta, strided_gamma, z, l, u
        )
        # print(l1.numpy(), u1.numpy())

        change_points = []
        for i in range(len(signs) - 1):
            if signs[i] == True and signs[i + 1] == False:
                change_points.append(i + 1)

        maximum_error = tf.gather(error_score, change_points)
        alpha_maximum = tf.gather(alpha, change_points)
        beta_maximum = tf.gather(beta, change_points)
        gamma_maximum = tf.gather(gamma, change_points)

        sort_index = tf.argsort(maximum_error, direction="DESCENDING")
        alpha_orderd = tf.gather(alpha_maximum, sort_index)
        beta_orderd = tf.gather(beta_maximum, sort_index)
        gamma_orderd = tf.gather(gamma_maximum, sort_index)

        alpha_orderd_diff = alpha_orderd[1:] - alpha_orderd[:-1]
        beta_orderd_diff = beta_orderd[1:] - beta_orderd[:-1]
        gamma_orderd_diff = gamma_orderd[1:] - gamma_orderd[:-1]

        # print(alpha_orderd_diff.numpy(), beta_orderd_diff.numpy(), gamma_orderd_diff.numpy())

        l, u = self._quadratics_to_interval(
            alpha_orderd_diff, beta_orderd_diff, gamma_orderd_diff, z, l, u
        )
        # print(l2.numpy(), u2.numpy())

        cps = set()
        for i in sort_index[: self.num_cp]:
            cps.add(change_points[i])

        return cps, [[l, u]]

    def model_selector(self, cps):
        return cps == self.cps

    def _make_feature_vectors(self, data, cps):
        pass

    def construct_eta(self, data):
        cps = self._detect_change_points(data)
        if cps is None:
            return None

        self.features = self._make_feature_vectors(data, cps)
        self.etas = []
        for i in range(1, len(self.features)):
            self.etas.append(self.features[i] - self.features[i - 1])
        self.cps = set(cps[1:-1])
        return self.etas


class ChangesForLinearTrend(ChangesBase):
    def __init__(
        self, model, look_back=10, predict_length=10, num_cp=2, smooth_window=5
    ):
        super().__init__(model, look_back, predict_length, num_cp, smooth_window)

    def _make_feature_vectors(self, data, cps):
        feature_vectors = []
        for j in range(1, len(cps)):
            cp_l = cps[j - 1]
            cp_r = cps[j]

            mask = tf.range(len(data))
            mask = tf.logical_and(mask > cp_l, mask <= cp_r)

            time = tf.range(len(data), dtype=tf.float64)[mask]
            feature = (time - tf.reduce_mean(time)) / (
                len(time) * tf.nn.moments(time, axes=0)[1]
            )
            feature = tf.tensor_scatter_nd_update(
                tf.zeros(len(data), dtype=tf.float64),
                tf.where(mask),
                feature,
            )
            feature_vectors.append(feature)
        return feature_vectors


class ChangesForMeanShift(ChangesBase):
    def __init__(
        self, model, look_back=10, predict_length=10, num_cp=2, smooth_window=5
    ):
        super().__init__(model, look_back, predict_length, num_cp, smooth_window)

    def _make_feature_vectors(self, data, cps):
        feature_vectors = []
        for j in range(1, len(cps)):
            cp_l = cps[j - 1]
            cp_r = cps[j]

            mask = tf.range(len(data))
            mask = tf.logical_and(mask > cp_l, mask <= cp_r)

            feature = tf.ones(cp_r - cp_l, dtype=tf.float64) / (cp_r - cp_l)
            feature = tf.tensor_scatter_nd_update(
                tf.zeros(len(data), dtype=tf.float64),
                tf.where(mask),
                feature,
            )
            feature_vectors.append(feature)
        return feature_vectors
