import tensorflow as tf
from tensorflow.keras import layers
from collections import OrderedDict

from .layers.update import BasicUpdateBlock, SmallUpdateBlock
from .layers.extractor import BasicEncoder, SmallEncoder
from .layers.corr import CorrBlock, bilinear_sampler, coords_grid, upflow8


class RAFT(tf.keras.Model):
    def __init__(self, drop_rate=0, iters=12, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        self.drop_rate = drop_rate

        self.iters = iters

        self.fnet = BasicEncoder(output_dim=256,
                                 norm_type='instance',
                                 drop_rate=drop_rate)
        self.cnet = BasicEncoder(output_dim=hdim+cdim,
                                 norm_type='batch',
                                 drop_rate=drop_rate)
        self.update_block = BasicUpdateBlock(filters=hdim)

    def initialize_flow(self, image):
        bs, h, w, _ = image.shape
        # shape: (bs, h/8, w/8, 2)x2, grid of batch/height/width
        coords0 = coords_grid(bs, h//8, w//8)
        coords1 = coords_grid(bs, h//8, w//8)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        ''' Upsample flow (h, w, 2) -> (8xh, 8xw, 2) using convex combination
        Args:
          flow: tensor with shape (bs, h, w, 2)
          mask: tensor with shape (bs, h, w, 64x9), 64=8x8 is the upscale
                9 is the neighborhood pixels in unfolding
        
        Returns:
          upscaled flow with shape (bs, 8xh, 8xw, 2)
        '''
        # flow: (bs, h, w, 2), mask: (bs, h, w, 64*9)
        bs, h, w, _ = flow.shape
        mask = tf.reshape(mask, (bs, h, w, 8, 8, 9, 1))
        mask = tf.nn.softmax(mask, axis=5)

        # flow: (bs, h, w, 2) -> (bs, h, w, 2*9)
        up_flow = tf.image.extract_patches(8*flow,
                                           sizes=(1, 3, 3, 1),
                                           strides=(1, 1, 1, 1),
                                           rates=(1, 1, 1, 1),
                                           padding='SAME')
        up_flow = tf.reshape(up_flow, (bs, h, w, 1, 1, 9, 2))
        # (bs, h, w, 8, 8, 9, 2) -> (bs, h, w, 8, 8, 2)
        up_flow = tf.reduce_sum(mask*up_flow, axis=5)
        # (bs, h, w, 8, 8, 2) -> (bs, h, w, 8x8x2)
        up_flow = tf.reshape(up_flow, (bs, h, w, -1))
        # (bs, h, w, 8x8x2) -> (bs, 8xh, 8xw, 2)
        return tf.nn.depth_to_space(up_flow, block_size=8)

    def call(self, inputs, training):
        image1, image2 = inputs
        image1 = 2*(image1/255.0) - 1.0
        image2 = 2*(image2/255.0) - 1.0

        # feature extractor -> (bs, h/8, w/8, 256)x2
        fmap1, fmap2 = self.fnet([image1, image2], training=training)

        # setup correlation values
        correlation = CorrBlock(fmap1, fmap2,
                                num_levels=self.corr_levels,
                                radius=self.corr_radius)

        # context network -> (bs, h/8, w/8, hdim+cdim)
        cnet = self.cnet(image1, training=training)
        # split -> (bs, h/8, w/8, hdim), (bs, h/8, w/8, cdim)
        net, inp = tf.split(cnet, [self.hidden_dim, self.context_dim], axis=-1)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        # (bs, h/8, w/8, 2)x2, xy-indexing
        coords0, coords1 = self.initialize_flow(image1)

        flow_predictions = []
        for i in range(self.iters):
            # (bs, h, w, 81xnum_levels)
            corr = correlation.retrieve(coords1)

            flow = coords1 - coords0
            # (bs, h, w, *), net: hdim, up_mask: 64x9, delta_flow: 2
            net, up_mask, delta_flow = self.update_block([net, inp, corr, flow])

            # F(t+1) = F(t) + df
            coords1 += delta_flow

            # upsample prediction
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        # flow_predictions[-1] is the finest output
        return flow_predictions

    def compile(self, optimizer, loss, epe, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.epe = epe

        self.flow_metrics = OrderedDict({
            'loss': tf.keras.metrics.Mean(name='loss'),
            'epe': tf.keras.metrics.Mean(name='epe'),
            'u1': tf.keras.metrics.Mean(name='u1'),
            'u3': tf.keras.metrics.Mean(name='u3'),
            'u5': tf.keras.metrics.Mean(name='u5')
        })

    def train_step(self, data):
        image1, image2, flow, valid = data
        image1 = tf.cast(image1, dtype=tf.float32)
        image2 = tf.cast(image2, dtype=tf.float32)

        with tf.GradientTape() as tape:
            flow_predictions = self([image1, image2], training=True)
            loss = self.loss([flow, valid], flow_predictions)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        info = self.epe([flow, valid], flow_predictions)
        self.flow_metrics['loss'].update_state(loss)
        self.flow_metrics['epe'].update_state(info['epe'])
        self.flow_metrics['u1'].update_state(info['u1'])
        self.flow_metrics['u3'].update_state(info['u3'])
        self.flow_metrics['u5'].update_state(info['u5'])
        return {k: m.result() for k, m in self.flow_metrics.items()}

    def reset_metrics(self):
        for k, m in self.flow_metrics.items():
            m.reset_states()


class SmallRAFT(RAFT):
    def __init__(self, drop_rate=0, iters=12, **kwargs):
        super().__init__(drop_rate, iters, **kwargs)

        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64
        self.corr_levels = 4
        self.corr_radius = 3

        self.fnet = SmallEncoder(output_dim=128,
                                 norm_type='instance',
                                 drop_rate=drop_rate)
        self.cnet = SmallEncoder(output_dim=hdim+cdim,
                                 norm_type=None,
                                 drop_rate=drop_rate)
        self.update_block = SmallUpdateBlock(filters=hdim)

    def call(self, inputs, training):
        image1, image2 = inputs
        image1 = 2*(image1/255.0) - 1.0
        image2 = 2*(image2/255.0) - 1.0

        # feature extractor -> (bs, h/8, w/8, nch)x2
        fmap1, fmap2 = self.fnet([image1, image2], training=training)

        # setup correlation values
        correlation = CorrBlock(fmap1, fmap2,
                                num_levels=self.corr_levels,
                                radius=self.corr_radius)

        # context network
        cnet = self.cnet(image1, training=training)
        net, inp = tf.split(cnet, [self.hidden_dim, self.context_dim], axis=-1)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        flow_predictions = []
        for i in range(self.iters):
            corr = correlation.retrieve(coords1)

            flow = coords1 - coords0
            net, _, delta_flow = self.update_block([net, inp, corr, flow])

            # F(t+1) = F(t) + df
            coords1 += delta_flow

            # upsample prediction
            flow_up = upflow8(coords1 - coords0)
            flow_predictions.append(flow_up)

        return flow_predictions        
