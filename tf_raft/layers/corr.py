import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def tfa_sampler(image, coords, mask=False):
    ''' Value sampler using tfa.image.resampler
    Args:
      image: tensor with shape (bs*h*w, h/i**2, w/i**2, 1)
      coords: coordinates tensor with shape (bs*h*w,, 2r+1, 2r+1, 2), xy-indexing
      mask: not implemented (same as the original implementation)
      
    Returns:
      sampled tensor with shape (bs*h*w, 2r+1, 2r+1, 1)

    None that tfa.image.resampler is decorated with @tf.function.
    Recursive use of this can be expensive and exessive.
    '''
    _, h, w, _ = image.shape
    # output: (bs*h*w, 2r+1, 2r+1, 1)
    image = tfa.image.resampler(image, coords)

    if mask:
        raise NotImplementedError("mask is not implemented for True")
    return image


def bilinear_sampler(image, coords):
    ''' Value sampler using tf.gather_nd
    Args:
      image: tensor with shape (bs*h*w, h/i**2, w/i**2, 1)
      coords: coordinates tensor with shape (bs*h*w,, 2r+1, 2r+1, 2), xy-indexing
      mask: not implemented (same as the original implementation)
      
    Returns:
      sampled tensor with shape (bs*h*w, 2r+1, 2r+1, 1)
    '''    
    _, h, w, _ = image.shape
    # -> (bs*h*w, 2r+1, 2r+1)x2
    gx, gy = tf.unstack(coords, axis=-1)
    gx = tf.clip_by_value(gx, 0, w-1)
    gy = tf.clip_by_value(gy, 0, h-1)

    # corners: (bs*h*w, 2r+1, 2r+1)x4
    gx0 = tf.floor(gx)
    gx1 = tf.math.ceil(gx)
    gy0 = tf.floor(gy)
    gy1 = tf.math.ceil(gy)

    # coordinates: (bs*h*w, 2r+1, 2r+1, 2)x4
    g00 = tf.stack([gy0, gx0], axis=-1)
    g01 = tf.stack([gy0, gx1], axis=-1)
    g10 = tf.stack([gy1, gx0], axis=-1)
    g11 = tf.stack([gy1, gx1], axis=-1)

    # coefficients: (bs*h*w, 2r+1, 2r+1, 1)x4
    c00 = tf.expand_dims((gy1 - gy)*(gx1 - gx), axis=-1)
    c01 = tf.expand_dims((gy1 - gy)*(gx - gx0), axis=-1)
    c10 = tf.expand_dims((gy - gy0)*(gx1 - gx), axis=-1)
    c11 = tf.expand_dims((gy - gy0)*(gx - gx0), axis=-1)

    # gathered values: (bs*h*w, 2r+1, 2r+1, 1)
    x00 = tf.gather_nd(image, tf.cast(g00, dtype=tf.int32), batch_dims=1)
    x01 = tf.gather_nd(image, tf.cast(g01, dtype=tf.int32), batch_dims=1)
    x10 = tf.gather_nd(image, tf.cast(g10, dtype=tf.int32), batch_dims=1)
    x11 = tf.gather_nd(image, tf.cast(g11, dtype=tf.int32), batch_dims=1)

    output = c00 * x00 + c01 * x01 + c10 * x10 + c11 * x11
    return output


def coords_grid(batch_size, height, width):
    ''' Generate coordinates (xy-order) from given info
    Args:
      batch_size, height, width: int values

    Returns:
      coordinates tensor with shape (bs, h, w, 2), xy-indexing
    '''
    # shape: (height, width)x2
    gy, gx = tf.meshgrid(tf.range(height, dtype=tf.float32),
                         tf.range(width, dtype=tf.float32),
                         indexing='ij')
    # -> (height, width, 2)
    coords = tf.stack([gx, gy], axis=-1)
    # -> (1, height, width, 2)
    coords = tf.expand_dims(coords, axis=0)
    # -> (batch_size, height, width, 2)
    coords = tf.tile(coords, (batch_size, 1, 1, 1))
    return coords


def upflow8(flow, mode='bilinear'):
    _, h, w, _ = flow.shape
    new_size = (8*h, 8*w)
    return 8*tf.image.resize(flow, new_size, mode)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_levels = num_levels
        self.radius = radius

        corr = self.correlation(fmap1, fmap2)
        batch_size, h1, w1, _, h2, w2 = corr.shape
        corr = tf.reshape(corr, (batch_size*h1*w1, h2, w2, 1))

        # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)

    def retrieve(self, coords):
        ''' Retrieve correlation values specified by coordinates
        Args:
          coords: coordinates tensor, shape (batch_size, h, w, 2)
        
        Returns:
          A tensor contains multiscale correlation
            with shape (bs, h, w, 81*num_levels)
        '''
        r = self.radius
        bs, h, w, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
            corr = self.corr_pyramid[i]
            # (2r+1, 2r+1)x2
            d = tf.range(-r, r+1, dtype=tf.float32)
            dy, dx = tf.meshgrid(d, d, indexing='ij')
            # (2r+1, 2r+1, 2)
            delta = tf.stack([dy, dx], axis=-1)
            # -> (1, 2r+1, 2r+1, 2)
            delta_lvl = tf.reshape(delta, (1, 2*r+1, 2*r+1, 2))

            # reshape and scale -> (bs*h*w, 1, 1, 2)
            centroid_lvl = tf.reshape(coords, (bs*h*w, 1, 1, 2)) / 2**i
            # add -> (bs*h*w, 2r+1, 2r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # output: (bs*h*w, 2r+1, 2r+1, dim)
            corr = bilinear_sampler(corr, coords_lvl)
            # -> (bs, h, w, (2r+1)*(2r+1)*nch)
            corr = tf.reshape(corr, (bs, h, w, -1))
            out_pyramid.append(corr)

        out = tf.concat(out_pyramid, axis=-1)
        return out

    def correlation(self, fmap1, fmap2):
        batch_size, h, w, nch = fmap1.shape
        fmap1 = tf.reshape(fmap1, (batch_size, h*w, nch))
        fmap2 = tf.reshape(fmap2, (batch_size, h*w, nch))

        # shape (batch_size, h*w, h*w)
        corr = tf.matmul(fmap1, fmap2, transpose_b=True)
        corr = tf.reshape(corr, (batch_size, h, w, 1, h, w))
        return corr / tf.sqrt(tf.cast(nch, dtype=tf.float32))        
