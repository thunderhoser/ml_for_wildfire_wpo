"""USE ONCE AND DESTROY."""

import os
import sys
import tensorflow.keras as tf_keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

from accum_optimizers import Optimizer

optimizer_function = Optimizer(
    optimizer=tf_keras.optimizers.Nadam(),
    steps=2
)
