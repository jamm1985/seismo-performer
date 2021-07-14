import tensorflow as tf
from keras.models import model_from_json


def load_model(model_path, weights_path):

    print('Keras loader call!')
    print('model_path = ', model_path)
    print('weights_path = ', weights_path)

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json, custom_objects = {"tf": tf})

    model.load_weights(weights_path)

    return model
