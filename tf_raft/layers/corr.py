import tensorflow as tf
from tensorflow.keras import layers


def bilinear_sampler(image, coords, mask=False):
    _, h, w, _ = image.shape

    # coords: (bs, h, w, 2r+1, 2r+1, 3) -> (bs, h, w, 2r+1, 2r+1)x3
    gb, gy, gx = tf.unstack(coords, axis=-1)
    gy = tf.clip_by_value(gy, 0, h-1)
    gx = tf.clip_by_value(gx, 0, w-1)
    coords = tf.stack([gb, gy, gx], axis=-1)

    # output: (bs, h, w, 2r+1, 2r+1, nch)
    image = tf.gather_nd(image, coords)
    if mask:
        raise NotImplementedError("mask is not implemented for True")
    return image


def coords_grid(batch_size, height, width):
    gb, gy, gx = tf.meshgrid(tf.range(batch_size),
                             tf.range(height),
                             tf.range(width),
                             indexing='ij')
    # shape: (bs, h, w, 3)
    coords = tf.stack([gb, gy, gx], axis=-1)
    return coords


# def coords_grid(batch_size, height, width):
#     #
#     r = 4
#     gb, gy, gx = tf.meshgrid(tf.range(batch_size),
#                              tf.range(height),
#                              tf.range(width),
#                              indexing='ij')
#     # gbyx: (bs, h, w, 3)
#     coords = tf.stack([gb, gy, gx], axis=-1)
#     # -> (bs, h, w, 1, 1, 3)
#     coords = tf.reshape(coords, (batch_size, height, width, 1, 1, 3))
#     # diff (2r+1, 2r+1)x2
#     dy, dx = tf.meshgrid(tf.linspace(-r, r, 2*r+1), tf.linspace(-r, r, 2*r+1),
#                          indexing='ij')
#     # -> (2r+1, 2r+1, 3)
#     gd = tf.stack([tf.zeros_like(dy), dy, dx], axis=-1)
#     # -> (1, 1, 1, 2r+1, 2r+1, 3)
#     gd = tf.reshape(gd, (1, 1, 1, 2*r+1, 2*r+1, 3))
#     gd = tf.cast(gd, tf.int32)
#     # add: (bs, h, w, 2r+1, 2r+1, 3)
#     coords += gd
    
#     # goal: (bs, h, w, 2r+1, 2r+1, 3)
#     return coords


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
        batch_size, h1, w1, dim, h2, w2 = corr.shape
        corr = tf.reshape(corr, (batch_size*h1*w1, h2, w2, dim))

        # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)

    def retrieve(self, coords):
        r = self.radius
        bs, h, w, _ = coords.shape
        # -> (bs, h, w, 1, 1, 3)
        coords = tf.reshape(coords, (batch_size, height, width, 1, 1, 3))
        # -> (bs, h, w, 1, 1, 1), (bs, h, w, 1, 1, 2)
        gb, gyx = tf.split(coords, (1, 2), axis=-1)

        out_pyramid = []
        for i in range(self.num_levels):
            # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
            corr = self.corr_pyramid[i]
            d = tf.linspace(-r, r, 2*r+1)
            # (2r+1, 2r+1)x2
            dy, dx = tf.meshgrid(d, d, indexing='ij')
            # (2r+1, 2r+1, 3)
            delta = tf.stack([tf.zeros_like(dy), dy, dx], axis=-1)
            # -> (1, 1, 1, 2r+1, 2r+1, 3)
            delta_lvl = tf.reshape(delta, (1, 1, 1, 2*r+1, 2*r+1, 3))

            # scale and concat -> (bs, h, w, 1, 1, 3)
            centroid_lvl = tf.concat([gb, gyx/2**i], axis=-1)
            # -> (bs, h, w, 2r+1, 2r+1, 3)
            coords_lvl = centroid_lvl + delta_lvl

            # output: (bs, h, w, 2r+1, 2r+1, nch)
            corr = bilinear_sampler(corr, coords_lvl)
            # -> (bs, h, w, (2r+1)*(2r+1)*nch)
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
