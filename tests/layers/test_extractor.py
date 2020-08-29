import pytest
import numpy as np
import tensorflow as tf


SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
IMAGE_SIZE = (128, 128)
NUM_CHANNELS = 32
FILTERS = 64
BATCH_SIZE = 16

