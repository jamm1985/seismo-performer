from tensorflow import keras
from tensorflow.keras import layers


def gpd(n_samples = 400, n_channels = 3, n_classes = 3, flatten = True):
    """
    This is ConvNet model definition from the "Generalized Seismic Phase Detection with Deep Learning"
    article by Zachary E. Ross, Men-Andrin Meier, Egill Hauksson, and Thomas H. Heaton.
    :param n_samples: Number of samples in a single input channel, input shape: (data_count, n_samples, n_channels)
    :param n_channels: Number of channels in input data, input shape: (data_count, n_samples, n_channels)
    :param n_classes: number of prediction classes, default: 3 (P-wave, S-wave and noise)
    :param flatten: True - flatten CNN output data in a single vector, False - use average pooling for every CNN
        filter, should work faster. In original paper CNN output is FLATTENED. Default: True.
    :return: keras model
    """
    inputs = layers.Input(shape=(n_samples, n_channels))

    # CNN
    n_filters = [32, 64, 128, 256]
    s_kernels = [21, 15, 11, 9]

    x = inputs
    for n_filter, s_kernel in zip(n_filters, s_kernels):

        x = layers.Conv1D(filters = n_filter, kernel_size = s_kernel, padding = 'same', activation = None)(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

    # Concatenate to a single vector
    if flatten:
        x = layers.Flatten()(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)

    # FCNN
    for _ in range(2):
        x = layers.Dense(200, activation = None)(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

    outputs = layers.Dense(n_classes, activation = 'softmax')(x)

    return keras.Model(inputs, outputs)


def load_model(weights_path):

    model = gpd()

    model.load_weights(weights_path)

    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = [keras.metrics.SparseCategoricalAccuracy()]
    )

    return model
