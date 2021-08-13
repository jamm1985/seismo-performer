from tensorflow import keras
import seismo_performer as sp


def load_performer(weights_path = None):
    """
    Loads fast-attention ST model variant.
    :param weights_path:
    :return:
    """
    _model = sp.seismo_performer_with_spec(
                                        maxlen=400,
                                        nfft=64,
                                        hop_length=16,
                                        patch_size_1=22,
                                        patch_size_2=3,
                                        num_channels=3,
                                        num_patches=11,
                                        d_model=48,
                                        num_heads=2,
                                        ff_dim_factor=2,
                                        layers_depth=2,
                                        num_classes=3,
                                        drop_out_rate=0.1)

    if weights_path is not None:
        _model.load_weights(weights_path)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()])

    return _model


def load_cnn(weights_path = None):
    """
    Loads CNN model on top of spectrogram.
    :param weights_path:
    :return:
    """
    _model = sp.model_cnn_spec(400,64,16)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()],)

    if weights_path is not None:
        _model.load_weights(weights_path)

    return _model
