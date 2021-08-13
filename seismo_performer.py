"""
File: seismo_performer.py
Author: Andrey Stepnov
Email: myjamm@gmail.com
Github: https://github.com/jamm1985
Description: model layers, model itself and auxiliary functions
"""

import math
import six
import h5py
from sklearn.model_selection import train_test_split
import itertools 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops.layers.tensorflow import Rearrange

from fast_attention import fast_attention

from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer

import numpy as np


def load_hdf5_to_numpy(filename):
    """
    Read hdf5 data file,
    load waveforms and labels to numpy array
    Parameters
    :filename: HDF5 file name
    :returns: tuple (X, Y) numpy arrays with samples and labels
    """
    f = h5py.File(filename, 'r')
    X = f['X']
    Y = f['Y']
    X_numpy = X[()]
    Y_numpy = Y[()]
    f.close()
    return X_numpy, Y_numpy


def load_test_train_data(hdf5_file, proportion, random_state=1):
    """
    Load data to numpy arrray from HDF5 
    and split to train and test sets with shuffle
    Parameters:
    :hdf5_file: string, path to HDF5 file
    :proportion: size of test set
    :random_state: fix state for testing purposes
    :returns: train and test sets with labels (numpy arrays)
    """
    # load data
    X, Y = load_hdf5_to_numpy(hdf5_file)
    # split dataset for train (75%), test (25%)
    print('Total samples {}'.format(X.shape[0]))
    count_y_values = np.unique(Y, return_counts=True)
    print('P {}, S {}, Noise {}'.format(
        count_y_values[1][0],
        count_y_values[1][1],
        count_y_values[1][2]))
    X_train, X_test, y_train, y_test\
        = train_test_split(
            X,
            Y,
            test_size=proportion,
            random_state=random_state,
            shuffle=True)
    # check for imbalance
    print(
        "test P, S and noise labels is {}%".format(
            np.unique(y_test,
                      return_counts=True)[1]/y_test.shape[0]))
    print(
        "train P, S and noise labels is {}%".format(
            np.unique(y_train,
                      return_counts=True)[1]/y_train.shape[0]))
    return X_train, X_test, y_train, y_test


"""
Learnable classification token
"""


class ClsToken(keras.layers.Layer):
    def __init__(self, embed_dim=20):
        super(ClsToken, self).__init__()
        self.embed_dim = embed_dim
        self.w = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32,
            trainable=True,
            name='ClsTokenW'
        )

    def call(self, inputs):
        self.batch_size = tf.shape(inputs)[0]
        self.x = tf.broadcast_to(
            self.w, [self.batch_size, 1, self.embed_dim]
        )
        return tf.keras.layers.concatenate([self.x, inputs], axis=1)


"""
Learnable position embedding
"""


class PosEmbeding(keras.layers.Layer):
    def __init__(self, num_patches=20, embed_dim=20):
        super(PosEmbeding, self).__init__()
        self.w = self.add_weight(
            shape=(num_patches, embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32,
            trainable=True,
            name='PosEmbedingW'
        )

    def call(self, inputs):
        return inputs + self.w


class PosEmbeding2(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PosEmbeding2, self).__init__()
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = inputs + self.position_embedding(positions)
        return encoded


"""
Rearrange 3 channels with patches to 1 channel
"""
class RearrangeCh(keras.layers.Layer):
    def __init__(self, num_patches=20, embed_dim=20):
        super(RearrangeCh, self).__init__()
        self.rearrange = Rearrange('b c n w -> b n (c w)')

    def call(self, inputs):
        return self.rearrange(inputs)


"""
Rearrange 3d channels 
"""
class Rearrange3d(keras.layers.Layer):
    def __init__(self, p1, p2):
        super(Rearrange3d, self).__init__()
        self.rearrange = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
                p1 = p1, p2 = p2)

    def call(self, inputs):
        return self.rearrange(inputs)


"""
Rescale to [0,1]
"""
class MMScaler(keras.layers.Layer):
    def __init__(self):
        super(MMScaler, self).__init__()

    def call(self, inputs):
        return (inputs - tf.reduce_min(inputs)) / (tf.reduce_max(inputs) - tf.reduce_min(inputs))



"""
Rescale to [-1,1]
"""
class MaxABSScaler(keras.layers.Layer):
    def __init__(self):
        super(MaxABSScaler, self).__init__()

    def call(self, inputs):
        min_abs_val = tf.abs(tf.reduce_min(inputs))
        max_abs_val = tf.abs(tf.reduce_max(inputs))
        max_abs = tf.maximum(min_abs_val, max_abs_val)
        return inputs / max_abs


"""
Implement a Performer block as a layer
"""
class PerformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(PerformerBlock, self).__init__()
        self.att = fast_attention.Attention(
            num_heads=num_heads, hidden_size=embed_dim, attention_dropout=0.1)
        self.ffn1 = layers.Dense(ff_dim, activation='gelu')
        self.ffn2 = layers.Dense(embed_dim, activation='gelu')
        self.add1 = layers.Add()
        self.add2 = layers.Add()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        ln_1 = self.layernorm1(inputs)
        attn_output = self.att(ln_1, ln_1, bias=None)
        add_1 = self.add1([attn_output, inputs])
        ln_2 = self.layernorm1(add_1)
        mlp_1 = self.ffn1(ln_2)
        dropout_1 = self.dropout1(mlp_1)
        mlp_2 = self.ffn2(dropout_1)
        dropout2 = self.dropout2(mlp_2)
        return self.add2([dropout2, add_1])



def seismo_performer_with_spec(
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
        drop_out_rate=0.1):
    """
    The model for P/S/N waves classification using ViT approach
    with converted raw signal to spectrogram and the treat it as input to PERFORMER
    Parameters:
    :maxlen: maximum samples of waveforms
    :nfft: number of FFTs in short-time Fourier transform
    :hop_length: Hop length in sample between analysis windows
    :patch_size_1: patch size for first dimention (depends on nfft/hop_length)
    :patch_size_2: patch size for second dimention (depends on nfft/hop_length)
    :num_channels: number of channels (usually it's equal to 3)
    :num_patches: resulting number of patches (FIX manual setup!)
    :d_model: Embedding size for each token
    :num_heads: Number of attention heads
    :ff_dim_factor: Hidden layer size in feed forward network inside transformer
                    ff_dim = d_model * ff_dim_factor
    :layers_depth: The number of transformer blocks
    :num_classes: The number of classes to predict
    :returns: Keras model object
    """
    num_patches = num_patches
    ff_dim = d_model * ff_dim_factor
    inputs = layers.Input(shape=(maxlen, num_channels))
    # do transform
    x = STFT(n_fft=nfft,
            window_name=None,
            pad_end=False,
            hop_length=hop_length,
            input_data_format='channels_last',
            output_data_format='channels_last',)(inputs)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)
    # custom normalization
    x = MaxABSScaler()(x)
    # patch the input channel
    x = Rearrange3d(p1=patch_size_1,p2=patch_size_2)(x)
    # embedding
    x = tf.keras.layers.Dense(d_model)(x)
    # add cls token
    x = ClsToken(d_model)(x)
    # positional embeddings
    x = PosEmbeding2(num_patches=num_patches + 1, projection_dim=d_model)(x)
    # encoder block
    for i in range(layers_depth):
        x = PerformerBlock(d_model, num_heads, ff_dim, rate=drop_out_rate)(x)
    # to MLP head
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    # MLP-head
    x = layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.Dense(d_model*ff_dim_factor, activation='gelu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.Dense(d_model, activation='gelu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_cnn_spec(timewindow, nfft, hop_length=4):
    """build very base CNN model on top of spectrogram.
    :returns: keras model object 
    """
    # std_dev_input = 0.001
    inputs = keras.Input(shape=(timewindow, 3))
    x = STFT(n_fft=nfft,
            window_name=None,
            pad_end=False,
            hop_length=hop_length,
            input_data_format='channels_last',
            output_data_format='channels_last',)(inputs)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)
    x = MaxABSScaler()(x)
    #x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (60,60)))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(80, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
