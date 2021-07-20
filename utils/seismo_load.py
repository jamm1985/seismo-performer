from tensorflow import keras
import seismo_transformer as st


def load_performer(weights_path):
    """
    Loads fast-attention ST model variant.
    :param weights_path:
    :return:
    """
    _model = st.seismo_performer_with_spec(
                                        maxlen=400,
                                        nfft=128,
                                        hop_length=16,
                                        patch_size_1=18,
                                        patch_size_2=13,
                                        num_channels=3,
                                        num_patches=5,
                                        d_model=96,
                                        num_heads=2,
                                        ff_dim_factor=2,
                                        layers_depth=2,
                                        num_classes=3,
                                        drop_out_rate=0.3)

    _model.load_weights(weights_path)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()])

    return _model


def load_performer_hpa(weights_path):
    """
    Loads fast-attention ST model variant with high accuracy
    :param weights_path:
    :return:
    """
    _model = st.seismo_performer_with_spec(
                                        maxlen=400,
                                        nfft=128,
                                        hop_length=1,
                                        patch_size_1=273,
                                        patch_size_2=1,
                                        num_channels=3,
                                        num_patches=65,
                                        d_model=96,
                                        num_heads=2,
                                        ff_dim_factor=2,
                                        layers_depth=2,
                                        num_classes=3,
                                        drop_out_rate=0.3)

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
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

    _model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()],)

    _model.load_weights(weights_path)

    return _model
