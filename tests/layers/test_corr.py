import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tf_raft.layers.corr import coords_grid, bilinear_sampler


SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
BATCH_SIZE = 4
IMAGE_SIZE = (32, 32)


def test_sampler():
    r = 4
    h, w = IMAGE_SIZE
    image = tf.random.normal((BATCH_SIZE*h*w, h, w, 1))
    coordsx = tf.random.uniform((BATCH_SIZE*h*w, 2*r+1, 2*r+1), 0, w-1)
    coordsy = tf.random.uniform((BATCH_SIZE*h*w, 2*r+1, 2*r+1), 0, h-1)
    coords = tf.stack([coordsx, coordsy], axis=-1)

    valid = tfa.image.resampler(image, coords)
    actual = bilinear_sampler(image, coords)

    np.testing.assert_allclose(actual.numpy(), valid.numpy(),
                               atol=1e-5, rtol=1e-5)
    
