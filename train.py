"""
File: train.py
Author: Andrey Stepnov
Email: myjamm@gmail.com
Github: https://github.com/jamm1985/seismo-transformer
Description: train examples on large dataset
"""

import tensorflow as tf
from tensorflow import keras
from seismo_transformer import load_test_train_data, seismo_transformer, seismo_transformer_with_spec, seismo_performer_hybrid, seismo_performer_with_spec, model_cnn_spec

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
    d_model=32,
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


model_with_spec = seismo_transformer_with_spec(
    maxlen=400,
    nfft=128,
    patch_size_1=35,
    patch_size_2=13,
    num_channels=3,
    num_patches=5,
    d_model=80,
    num_heads=8,
    ff_dim_factor=4,
    layers_depth=1,
    num_classes=3,
    drop_out_rate=0.1)

LR = 0.001
model_with_spec.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)

print("Fit model on training data")
history = model.fit(
    X_train,
    epochs=20,
    validation_data=(X_test),
)


model_with_spec = seismo_performer_with_spec(
    maxlen=400,
    nfft=128,
    patch_size_1=5,
    patch_size_2=5,
    num_channels=3,
    num_patches=91,
    d_model=72,
    num_heads=8,
    ff_dim_factor=4,
    layers_depth=1,
    num_classes=3,
    drop_out_rate=0.1)

LR = 0.001
model_with_spec.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)


# load Dagestan test data M1.0+ fixed channel order
X_train, X_test, y_train, y_test =\
    load_test_train_data('/Volumes/ML_SEISMO_D/ML_DATASETS_SAKH_DAG/2020-09-10-DAGESTAN_1_FIXED.hdf5', 0.3)

# load Sakhalin test data M1.0+ fixed channel order
X_train, X_test, y_train, y_test =\
    load_test_train_data('/Volumes/ML_SEISMO_D/ML_DATASETS_SAKH_DAG/2020-09-10-SAKHALIN_1_FIXED.hdf5', 0.3)

model_with_spec.load_weights('/Users/jamm/Downloads/web/weights_model_performer_with_spec.96K..CALI.V1.hd5')

print("Fit model on training data")
history = model_with_spec.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_test, y_test))


model_with_spec.save('/Users/jamm/Downloads/web/mymodel')

model_hybrid = seismo_performer_hybrid(
        maxlen=400,
        patch_size=25,
        patch_size_1=9,
        patch_size_2=5,
        num_channels=3,
        num_patches_spec = 13,
        nfft=128,
        d_model=48,
        num_heads=8,
        ff_dim_factor=4,
        layers_depth=8,
        num_classes=3,
        drop_out_rate=0.1)
