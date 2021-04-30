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
# Learnable classification token
"""


class ClsToken(keras.layers.Layer):
    def __init__(self, embed_dim=20):
        super(ClsToken, self).__init__()
        self.embed_dim = embed_dim
        self.w = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs):
        self.batch_size = tf.shape(inputs)[0]
        self.x = tf.broadcast_to(
            self.w, [self.batch_size, 1, self.embed_dim]
        )
        return tf.keras.layers.concatenate([self.x, inputs], axis=1)


"""
# Learnable position embedding
"""


class PosEmbeding(keras.layers.Layer):
    def __init__(self, num_patches=20, embed_dim=20):
        super(PosEmbeding, self).__init__()
        self.w = self.add_weight(
            shape=(num_patches, embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs):
        return inputs + self.w


# Rearrange 3 chanles with patches to 1 channel
class RearrangeCh(keras.layers.Layer):
    def __init__(self, num_patches=20, embed_dim=20):
        super(RearrangeCh, self).__init__()
        self.rearrange = Rearrange('b c n w -> b n (c w)')

    def call(self, inputs):
        return self.rearrange(inputs)


"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=rate)
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

