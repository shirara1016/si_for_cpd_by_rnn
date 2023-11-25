import tensorflow as tf


class ReconstructErrorMap(tf.keras.layers.Layer):
    def __init__(self, mode="thr", thr=0.1, k=20):
        super().__init__()
        self.mode = mode
        self.thr = thr
        self.k = k

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "thr": self.thr, "k": self.k})
        return config

    def get_weights(self):
        return [None]

    def call(self, inputs):
        input = inputs[0]
        predict = inputs[1]
        if self.mode == "thr":
            error = tf.abs(input - predict)
            mapping = tf.cast(error > self.thr, dtype=tf.int64)
        elif self.mode == "top-k":
            pass
        else:
            raise Exception("The mode is invalid.")
        return mapping

    def compute_output_shape(self, input_shape):
        return input_shape


def rnn_saliency_map(
    input_dim=1, hidden_dim=32, output_dim=1, mode="thr", thr=0.1, k=20
):
    inputs = tf.keras.Input(shape=(None, input_dim))
    rnn = tf.keras.layers.SimpleRNN(
        hidden_dim, return_sequences=True, activation="relu"
    )(inputs)
    predicts = tf.keras.layers.Dense(output_dim)(rnn)
    mapping = ReconstructErrorMap(mode=mode, thr=thr, k=k)([inputs, predicts])
    model = tf.keras.Model(inputs, mapping)
    return model


def rnn_predict_model(input_dim=1, hidden_dim=32, output_dim=1):
    inputs = tf.keras.Input(shape=(None, input_dim))
    rnn = tf.keras.layers.SimpleRNN(
        hidden_dim, return_sequences=True, activation="relu"
    )(inputs)
    predicts = tf.keras.layers.Dense(output_dim)(rnn)
    model = tf.keras.Model(inputs, predicts)
    return model


def rnn_labeling_model(input_dim=1, hidden_dim=32, output_dim=1):
    inputs = tf.keras.Input(shape=(None, input_dim))
    rnn = tf.keras.layers.SimpleRNN(
        hidden_dim, return_sequences=True, activation="relu"
    )(inputs)
    predicts = tf.keras.layers.Dense(output_dim, activation="sigmoid")(rnn)
    model = tf.keras.Model(inputs, predicts)
    return model


def one_step_predict_model(input_dim=1, hidden_dim=8, output_dim=1):
    inputs = tf.keras.layers.Input(shape=(None, input_dim))
    rnn = tf.keras.layers.SimpleRNN(hidden_dim, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(output_dim)(rnn)
    model = tf.keras.Model(inputs, outputs)
    return model
