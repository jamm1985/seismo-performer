"""
File: train.py
Author: Andrey Stepnov
Email: myjamm@gmail.com
Github: https://github.com/jamm1985
Description: train model on large dataset
"""

import tensorflow as tf
from tensorflow import keras
from seismo_transformer import load_test_train_data, seismo_transformer

import sys

from h5_generator import train_test_split as h5_tts

X_train, _ = h5_tts('/home/vchernyh/jamm/DATA/2020-09-10-SAKHALIN_1_FIXED.hdf5',\
                    batch_size = 500,\
                    test_size = 0.1, random_state = 52)

_, X_test = h5_tts('data/2019_2021_normalized.h5',\
                    batch_size = 50,\
                    test_size = 0.99, random_state = 52)

# 'data/2019_2021_normalized.h5'
# 'data/2019_normalized.h5'
# 'data/seisan_current_normalized.h5'
# 'data/seisan_current_channeled.h5'
# 'data/seisan_1_current.h5'
# 'data/data_picker_normalized.h5'
# '/home/vchernyh/jamm/DATA/scsn_ps_2000_2017_shuf.hdf5'
# '/home/vchernyh/jamm/DATA/2020-09-10-SAKHALIN_1_FIXED.hdf5'
# 'data/data_picker_only_P.h5'

# y2_test = y2_test.astype('int')

# 'data/data_2019_seisan_1_fixed_final.h5'

# Only P, S or N ?

# phase = 0
# X2_test = X2_test[y2_test[:] == phase]
# y2_test = y2_test[y2_test[:] == phase]

# y2_test[:] = 2

# print(f'SIZE X: {len(X_train)}, SIZE X2: {len(X2_test)}')

# sys.exit(0)

# Convert to tf dataset
#X_train = tf.data.Dataset.from_tensor_slices(
#    (X_train, y_train)).batch(400)
#X2_test = tf.data.Dataset.from_tensor_slices(
#    (X2_test, y2_test)).batch(10)

model = seismo_transformer(
    maxlen=400,
    patch_size=25,
    num_channels=3,
    d_model=48,
    num_heads=8,
    ff_dim_factor=4,
    layers_depth=8,
    num_classes=3,
    drop_out_rate=0.1)

model.load_weights('WEIGHTS/model.240K.V1.h5')

LR = 0.001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],)

# model.load_weights('WEIGHTS/model.240K.V1.h5')

# print(f'SIZE X: {len(X_train)}, SIZE X2: {len(X2_test)}')

print("Fit model on training data")
history = model.fit(
    X_train,
    epochs=12,
    validation_data=(X_test),
)

scores = model.evaluate(X_test)

print('SCORES: ', scores)

# model.save_weights('WEIGHTS/sac_new_only.h5', save_format='h5')
