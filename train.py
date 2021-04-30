"""
File: train.py
Author: Andrey Stepnov
Email: myjamm@gmail.com
Github: https://github.com/jamm1985/seismo-transformer
Description: train examples on large dataset
"""

import tensorflow as tf
from tensorflow import keras
from seismo_transformer import load_test_train_data, seismo_transformer

# Load CalTech data to NUMPY array.
# takes 21GB of memory!!!
X_train, X_test, y_train, y_test\
    = load_test_train_data('DATA/scsn_ps_2000_2017_shuf.hdf5', 0.1)
# Convert to tf dataset
X_train = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).batch(480)
X_test = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(480)

model = seismo_transformer(
    maxlen=400,
    patch_size=25,
    num_channels=3,
    d_model=64,
    num_heads=8,
    ff_dim_factor=4,
    layers_depth=8,
    num_classes=3,
    drop_out_rate=0.1)

LR = 0.001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)

print("Fit model on training data")
history = model.fit(
    X_train,
    epochs=20,
    validation_data=(X_test),
)

model.save_weights('WEIGHTS/model.240K.V2.hd5', save_format='h5')
