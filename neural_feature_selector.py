import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Concatenate, BatchNormalization, Input
from tensorflow.keras.regularizers import L1, L2


def aggregating_cnn(inputs, n_filters):
    return Conv1D(filters=n_filters,
                  kernel_size=1,
                  padding="same",
                  strides=1,
                  dilation_rate=1,
                  activation="relu",
                  kernel_regularizer=L1(0.01))(inputs)


def tcnn_block(inputs, kernel_sizes=None):
    """
    Applies Conv1D to a feature in the input MTS (a feature is just a univariant TS)

    Args:
        inputs: A Tensor representing a MTS
        kernel_sizes: The Kernel Sizes of each Conv1D (there will be only one filter for each kernel_size)

    Returns:
        The encoded features
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
              activation="relu",
              kernel_regularizer=L2(0.01))(inputs) for x in kernel_sizes])
    x = aggregating_cnn(x, n_filters=1)
    return x


def tcnn_layer(inputs, n_features):
    """
    The TCNN layer applies a `tcnn_block` to each feature and then concatenates the output tensors
    by the last axis.

    Args:
        inputs: A MTS
        n_features: The number of features

    Returns:

    """
    return Concatenate(axis=-1)([tcnn_block(inputs[..., x]) for x in range(n_features)])


def nfs(inputs, input_shape, n_output_filters):
    """

    Args:
        inputs: A MTS
        input_shape: MTS input shape
        n_output_filters: The number of output filters to apply in the aggregating cnn

    Returns:
        A new MTS after Neural Feature Selector is applied. This MTS will be the output for
        other architectures: LSTM, Resnet, ...
    """
    _, m = input_shape

    x = tcnn_layer(inputs, m)
    x = BatchNormalization()(x)
    return aggregating_cnn(x, n_filters=n_output_filters)
