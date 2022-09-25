import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Concatenate, BatchNormalization, Input


def tcnn_block(inputs, kernel_sizes=None):
    """
    By default it applies four Conv1D per feature / channel of an MTS. Then,
    it simply concatenates the outputs.
    """
    if kernel_sizes is None:
        kernel_sizes = [2, 3, 4, 5]

    inputs = tf.expand_dims(inputs, axis=-1)
    x = Concatenate(axis=-1)(
      [Conv1D(filters=1,
              kernel_size=x,
              padding="same",
              strides=1,
              dilation_rate=1,
              activation="relu")(inputs) for x in kernel_sizes])
    x = aggregating_cnn(x, n_filters=1)
    return x


def tcnn_layer(inputs):
    return Concatenate(axis=-1)([tcnn_block(inputs[..., x]) for x in range(N_FEATURES)])


def aggregating_cnn(inputs, n_filters):
    return Conv1D(filters=n_filters,
                  kernel_size=1,
                  padding="same",
                  strides=1,
                  dilation_rate=1,
                  activation="relu")(inputs)


def nfs(inputs):
    x = tcnn_layer(inputs)
    x = BatchNormalization()(x)
    return aggregating_cnn(x, n_filters=OUTPUT_DIM)