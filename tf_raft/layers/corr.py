import tensorflow as tf
from tensorflow.keras import layers


def bilinear_sampler(image, coords, mask=False):
    _, h, w, _ = image.shape
    xgrid, ygrid = tf.split(coords, 2, axis=-1)

    # (bs*h*w, 2r+1, 2r+1)
    # (bs, h, w, 2r+1, 2r+1, 3)
    # -> (bs, h, w, 2r+1, 2r+1, nch)
    gh = tf.range(h)
    gw = tf.range(w)
    gr = tf.linspace(-r, r, 2*r+1)
    gy = tf.reshape(gh, (-1, 1)) + gr # (h, 2r+1)
    gx = tf.reshape(gw, (-1, 1)) + gr # (w, 2r+1)
    pass


def coords_grid(batch_size, height, width):
    #
    r = 4
    gb, gy, gx = tf.meshgrid(tf.range(batch_size),
                             tf.range(height),
                             tf.range(width),
                             indexing='ij')
    # gbyx: (bs, h, w, 3)
    coords = tf.stack([gb, gy, gx], axis=-1)
    # -> (bs, h, w, 1, 1, 3)
    coords = tf.reshape(coords, (batch_size, height, width, 1, 1, 3))
    # diff (2r+1, 2r+1)x2
    dy, dx = tf.meshgrid(tf.linspace(-r, r, 2*r+1), tf.linspace(-r, r, 2*r+1),
                         indexing='ij')
    # -> (2r+1, 2r+1, 3)
    gd = tf.stack([tf.zeros_like(dy), dy, dx], axis=-1)
    # -> (1, 1, 1, 2r+1, 2r+1, 3)
    gd = tf.reshape(gd, (1, 1, 1, 2*r+1, 2*r+1, 3))
    gd = tf.cast(gd, tf.int32)
    # add: (bs, h, w, 2r+1, 2r+1, 3)
    coords += gd
    
    # goal: (bs, h, w, 2r+1, 2r+1, 3)
    return coords
    
    


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_levels = num_levels
        self.radius = radius

        corr = self.correlation(fmap1, fmap2)
        batch_size, h1, w1, dim, h2, w2 = corr.shape
        corr = tf.reshape(corr, (batch_size*h1*w1, h2, w2, dim))

        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)

    def retrieve(self, coords):
        r = self.radius
        bs, h, w, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            d = tf.linspace(-r, r, 2*r+1)
            # delta = tf.stack(tf.meshgrid(d, d, indexing='ij'), axis=-1)
            delta = tf.stack(tf.meshgrid(d, d, indexing='xy'), axis=-1)

            centroid_lvl = tf.reshape(coords, (bs*h*w, 1, 1, 2)) / 2**i
            delta_lvl = tf.reshape(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = tf.reshape(bs, h, w, -1)
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
