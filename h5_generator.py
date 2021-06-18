from tensorflow.keras.utils import Sequence
import h5py as h5
import numpy as np
import math

"""
Features H5Generator which can be used in keras model.fit method for batching large h5 files and
train_test_split which returns train and test H5Generators.

Note that for only test or train datasets without shuffling H5Generator will work faster.

Usage example for only train or test set:

from h5_generator import H5Generator
X_train = H5Generator(path_to_file, 480)
# Load model here
history = model.fit(X_train, epochs = 100)

Usage example for train/test split:

from h5_generator import train_test_split as h5_tts
X_train, X_test = h5_tts(path_to_file, batch_size = 480, test_size = 0.2, random_state = 42)
# Load model here
history = model.fit(X_train, epochs = 100, validation_data = (X_test))
"""


class H5Generator(Sequence):

    def __init__(self, path, batch_size, x_name = 'X', y_name = 'Y',
                 idxs = None, length = None, start_pos = 0):
        
        self.h5_file = path
        self.x_name = 'X'
        self.y_name = 'Y'

        self.batch_size = batch_size

        self.idxs = idxs

        self.length = length
        self.start_pos = start_pos

    def  __len__(self):
        
        if self.idxs is not None:

            l = self.idxs.shape[0]

        else:

            if not self.length:
                with h5.File(self.h5_file) as f:
                    self.length = f[self.y_name].shape[0]

            l = self.length

        return math.ceil(l / self.batch_size)

    def __getitem__(self, idx):

        with h5.File(self.h5_file, 'r') as f:
        
            X, Y = (f[self.x_name], f[self.y_name])

            if self.idxs is not None:
                
                batch_idxs = list(np.sort(self.idxs[idx * self.batch_size : (idx + 1) * self.batch_size]))

                batch_x = X[batch_idxs]
                batch_y = Y[batch_idxs]

            else:

               batch_x = X[self.start_pos + idx * self.batch_size : self.start_pos + (idx + 1) * self.batch_size]
               batch_y = Y[self.start_pos + idx * self.batch_size : self.start_pos + (idx + 1) * self.batch_size]

        return batch_x, batch_y


def train_test_split(path, batch_size, x_name = 'X', y_name = 'Y',
                     test_size = None, train_size = None,
                     random_state = None, shuffle = True):
    """
    Returns H5Generator objects for train/test split.
    Positional arguments are made to replicate sklearn.model_selection.train_test_split arguments behaviour.
    Note, that for only test or train generator without shuffling H5Generator(...) is more preferable than
        x, _, y, _ = train_test_split(...), due to faster data reading.
    
    Arguments:

        path - path to h5 file.
        batch_size - batch size.
        x_name - name of X dataset in h5 file, default; "X".
        y_name - name of Y dataset in h5 file, default: "Y".

        test_size - [0, 1] - test split ratio, default: None.
        train_size - [0, 1] - train split ratio, default: None.
            If both test_size and train_size are None, then test_size will default to 0.25.

        random_state - random state for numpy.random.seed().
        shuffle - shuffle data? Default: True.
    """
    if random_state:
        np.random.seed(random_state)

    idxs = None
    data_length = 0
    if shuffle:
        with h5.File(path, 'r') as f:
            idxs = np.arange(f[y_name].shape[0])
        np.random.shuffle(idxs)
        data_length = idxs.shape[0]
    else:
        with h5.File(path, 'r') as f:
            data_length = f[y_name].shape[0]

    # Split
    r_train_size = None
    if test_size is None and train_size is None:
        r_test_size = 0.25
    elif test_size is None:
        r_test_size = 1. - train_size
    elif train_size is None:
        r_test_size = test_size
    else:
        r_test_size = test_size
        r_train_size = train_size

    if idxs is not None:

        if r_train_size is None:

            test_pos = math.ceil(idxs.shape[0] * r_test_size)

            test_idxs = idxs[:test_pos]
            train_idxs = idxs[test_pos:]

        else:

            if r_test_size + r_train_size > 1.:
                raise ValueError('test_size and train_size parameters are invalid!')

            test_pos = math.ceil(idxs.shape[0] * r_test_size)
            train_pos = math.floor(idxs.shape[0] * r_train_size)

            test_idxs = idxs[:test_pos]
            train_idxs = idxs[test_pos : test_pos + train_pos]

        # Create datasets
        X_train = H5Generator(path, batch_size, x_name, y_name, train_idxs)
        X_test = H5Generator(path, batch_size, x_name, y_name, test_idxs)

        return X_train, X_test

    else:

        if r_train_size is None:

            test_size = math.ceil(data_length * r_test_size)
            train_size = data_length - test_size

        else:

            if r_test_size + r_train_size > 1.:
                raise ValueError('test_size and train_size parameters are invalid!')

            test_size = math.ceil(data_length * r_test_size)
            train_size = math.floor(data_length * r_train_size)

        # Create datasets
        X_train = H5Generator(path, batch_size, x_name, y_name,
                              start_pos = test_size, length = train_size)
        X_test = H5Generator(path, batch_size, x_name, y_name,
                             start_pos = 0, length = test_size)

        return X_train, X_test
