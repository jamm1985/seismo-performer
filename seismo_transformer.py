"""
File: seismo_transformer.py
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
    """Read hdf5 data file,
    load seismological waveforms and labels and convert to numpy array
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
    """load data to numpy arrray from HDF5 
    and split to train and test sets with shuffle
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
Implement a Transformer block as a layer
Credit:
    Title: Text classification with Transformer
    Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim,
                activation='gelu'),
                layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PerformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(PerformerBlock, self).__init__()
        self.att = fast_attention.Attention(
            num_heads=num_heads, hidden_size=embed_dim, attention_dropout=0.0)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim,
                activation='gelu'),
                layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, bias=None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

"""
Models section
""" 
def seismo_transformer(
        maxlen=400,
        patch_size=25,
        num_channels=3,
        d_model=96,
        num_heads=8,
        ff_dim_factor=4,
        layers_depth=8,
        num_classes=3,
        drop_out_rate=0.1):
    """Model for P/S/N waves classification using ViT approach
    :maxlen: maximum samples of waveforms
    :patch_size: patch size for every single channel
    :num_channels: number of channels (usually it's equal to 3)
    :d_model: Embedding size for each token
    :num_heads: Number of attention heads
    :ff_dim_factor: Hidden layer size in feed forward network inside transformer
                    ff_dim = d_model * ff_dim_factor
    :layers_depth: The number of transformer blocks
    :num_classes: The number of classes to predict
    :returns: Keras model object
    """
    num_patches = maxlen // patch_size
    ff_dim = d_model * ff_dim_factor
    inputs = layers.Input(shape=(maxlen, num_channels))
    x = tf.keras.layers.Permute((2, 1))(inputs)
    # patch the input channel
    x = tf.keras.layers.Reshape((num_channels, num_patches, patch_size))(x)
    x = RearrangeCh()(x)
    # embedding
    x = tf.keras.layers.Dense(d_model)(x)
    # cls token
    x = ClsToken(d_model)(x)
    # positional embeddings
    x = PosEmbeding(num_patches=num_patches + 1, embed_dim=d_model)(x)
    # encoder block
    x = layers.Dropout(drop_out_rate)(x)
    for i in range(layers_depth):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x)
    # to MLP head
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    # MLP-head
    #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.Dense(ff_dim, activation='gelu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def seismo_transformer_with_spec(
        maxlen=400,
        patch_size_1=2,
        patch_size_2=3,
        num_channels=3,
        num_patches = 40,
        nfft=128,
        d_model=96,
        num_heads=8,
        ff_dim_factor=4,
        layers_depth=8,
        num_classes=3,
        drop_out_rate=0.1):
    """The model for P/S/N waves classification using ViT approach
    with converted raw to spectrogram input
    :maxlen: maximum samples of waveforms
    :patch_size_1: patch size for first dimention
    :patch_size_2: patch size for second dimention
    :num_channels: number of channels (usually it's equal to 3)
    :num_patches: resulting number of patches (FIX manual setup!)
    :nfft: number of FFTs in short-time Fourier transform
    :d_model: Embedding size for each token
    :num_heads: Number of attention heads
    :ff_dim_factor: Hidden layer size in feed forward network inside transformer
                    ff_dim = d_model * ff_dim_factor
    :layers_depth: The number of transformer blocks
    :num_classes: The number of classes to predict
    :returns: Keras model object
    """
    #num_patches = (maxlen // patch_size)**2
    num_patches = num_patches
    ff_dim = d_model * ff_dim_factor
    inputs = layers.Input(shape=(maxlen, num_channels))
    # x = tf.keras.layers.Permute((2, 1))(inputs)
    # do transform
    x = STFT(n_fft=nfft,
            window_name=None,
            pad_end=False,
            input_data_format='channels_last',
            output_data_format='channels_last',)(inputs)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)
    # patch the input channel
    # x = tf.keras.layers.Reshape((num_channels, num_patches, patch_size))(x)
    x = Rearrange3d(p1=patch_size_1,p2=patch_size_2)(x)
    # embedding
    x = tf.keras.layers.Dense(d_model)(x)
    # cls token
    x = ClsToken(d_model)(x)
    # positional embeddings
    x = PosEmbeding(num_patches=num_patches + 1, embed_dim=d_model)(x)
    # encoder block
    x = layers.Dropout(drop_out_rate)(x)
    for i in range(layers_depth):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x)
    # to MLP head
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    # MLP-head
    #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.Dense(ff_dim, activation='gelu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def seismo_performer_with_spec(
        maxlen=400,
        patch_size_1=2,
        patch_size_2=3,
        num_channels=3,
        num_patches = 40,
        nfft=128,
        d_model=96,
        num_heads=8,
        ff_dim_factor=4,
        layers_depth=8,
        num_classes=3,
        drop_out_rate=0.1):
    """The model for P/S/N waves classification using ViT approach
    with converted raw to spectrogram input
    :maxlen: maximum samples of waveforms
    :patch_size_1: patch size for first dimention
    :patch_size_2: patch size for second dimention
    :num_channels: number of channels (usually it's equal to 3)
    :num_patches: resulting number of patches (FIX manual setup!)
    :nfft: number of FFTs in short-time Fourier transform
    :d_model: Embedding size for each token
    :num_heads: Number of attention heads
    :ff_dim_factor: Hidden layer size in feed forward network inside transformer
                    ff_dim = d_model * ff_dim_factor
    :layers_depth: The number of transformer blocks
    :num_classes: The number of classes to predict
    :returns: Keras model object
    """
    #num_patches = (maxlen // patch_size)**2
    num_patches = num_patches
    ff_dim = d_model * ff_dim_factor
    inputs = layers.Input(shape=(maxlen, num_channels))
    # x = tf.keras.layers.Permute((2, 1))(inputs)
    # do transform
    x = STFT(n_fft=nfft,
            window_name=None,
            pad_end=False,
            input_data_format='channels_last',
            output_data_format='channels_last',)(inputs)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)
    # patch the input channel
    # x = tf.keras.layers.Reshape((num_channels, num_patches, patch_size))(x)
    x = Rearrange3d(p1=patch_size_1,p2=patch_size_2)(x)
    # embedding
    x = tf.keras.layers.Dense(d_model)(x)
    # cls token
    x = ClsToken(d_model)(x)
    # positional embeddings
    x = PosEmbeding(num_patches=num_patches + 1, embed_dim=d_model)(x)
    # encoder block
    x = layers.Dropout(drop_out_rate)(x)
    for i in range(layers_depth):
        x = PerformerBlock(d_model, num_heads, ff_dim)(x)
    # to MLP head
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    # MLP-head
    #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.Dense(ff_dim, activation='gelu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def seismo_transformer_hybrid(
        maxlen=400,
        patch_size=25,
        patch_size_1=2,
        patch_size_2=3,
        num_channels=3,
        num_patches_spec = 40,
        nfft=128,
        d_model=96,
        num_heads=8,
        ff_dim_factor=4,
        layers_depth=8,
        num_classes=3,
        drop_out_rate=0.1):
    """Create classifier model using ViT approach with transformer blocks
    :maxlen: maximum samples of waveforms
    :patch_size: patch size for every single channel
    :num_channels: number of channels (usually it's equal to 3)
    :d_model: Embedding size for each token
    :num_heads: Number of attention heads
    :ff_dim_factor: Hidden layer size in feed forward network inside transformer
                    ff_dim = d_model * ff_dim_factor
    :layers_depth: The number of transformer blocks
    :num_classes: The number of classes to predict
    :returns: Keras model object
    """
    num_patches = maxlen // patch_size
    ff_dim = d_model * ff_dim_factor
    inputs = layers.Input(shape=(maxlen, num_channels))
    x = tf.keras.layers.Permute((2, 1))(inputs)
    # patch the input channel
    x = tf.keras.layers.Reshape((num_channels, num_patches, patch_size))(x)
    x = RearrangeCh()(x)
    # embedding
    x = tf.keras.layers.Dense(d_model)(x)
    # do transform
    num_patches_spec = num_patches_spec
    spec = STFT(n_fft=nfft,
            window_name=None,
            pad_end=False,
            input_data_format='channels_last',
            output_data_format='channels_last',)(inputs)
    spec = Magnitude()(spec)
    spec = MagnitudeToDecibel()(spec)
    # patch the input channel
    # x = tf.keras.layers.Reshape((num_channels, num_patches, patch_size))(x)
    spec = Rearrange3d(p1=patch_size_1,p2=patch_size_2)(spec)
    # embedding
    spec = tf.keras.layers.Dense(d_model)(spec)
    # concatenate spec and patched raw input
    x = tf.keras.layers.Concatenate(axis=1)([x, spec])
    # cls token
    x = ClsToken(d_model)(x)
    # positional embeddings
    x = PosEmbeding(num_patches=num_patches+num_patches_spec + 1, embed_dim=d_model)(x)
    # encoder block
    x = layers.Dropout(drop_out_rate)(x)
    for i in range(layers_depth):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x)
    # to MLP head
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    # MLP-head
    #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = tf.keras.layers.Dense(ff_dim, activation='gelu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
