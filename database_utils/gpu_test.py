from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

import os

print(os.listdir("./"))
