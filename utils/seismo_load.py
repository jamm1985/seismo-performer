from tensorflow import keras
import seismo_transformer as st


def load_transformer(weights_path):
    """
    Loads standard ST model.
    :param weights_path: Path to weights file.
    :return:
    """
    _model = st.seismo_transformer(maxlen = 400,
                                   patch_size = 25,
                                   num_channels = 3,
                                   d_model = 48,
                                   num_heads = 8,
                                   ff_dim_factor = 4,
                                   layers_depth = 8,
                                   num_classes = 3,
                                   drop_out_rate = 0.1)

    _model.load_weights(weights_path)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()])

    return _model


def load_favor(weights_path):
    """
    Loads fast-attention ST model variant.
    :param weights_path:
    :return:
    """
    _model = st.seismo_performer_with_spec(
                                        maxlen=400,
                                        nfft=128,
                                        hop_length=4,
                                        patch_size_1=69,
                                        patch_size_2=13,
                                        num_channels=3,
                                        num_patches=5,
                                        d_model=48,
                                        num_heads=4,
                                        ff_dim_factor=4,
                                        layers_depth=2,
                                        num_classes=3,
                                        drop_out_rate=0.1)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()],)

    _model.load_weights(weights_path)

    return _model


def load_cnn(weights_path):
    """
    Loads CNN model on top of spectrogram.
    :param weights_path:
    :return:
    """
    _model = st.model_cnn_spec(400,128)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()],)

    _model.load_weights(weights_path)

    return _model
