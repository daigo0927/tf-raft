import os
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from .datasets.flow_viz import flow_to_image


def first_cycle_scaler(cycle):
    ''' Return learning rate scale 
    |--    cycle = 1    --|-- cycle > 1 --
    min_lr -> max_lr -> min_lr -> (const)
    '''
    return tf.cond(cycle == 1, lambda: 1.0, lambda: 0.0)


def inverse_scaler(cycle):
    ''' Return learning rate scale
    |--    cycle = 1    --|--    cycle = 2    --|--   cycle = 3 --
    min_lr -> max_lr -> min_lr -> max_lr/2 -> min_lr -> max_lr/3 -> ...
    '''
    return 1/cycle


class VisFlowCallback(callbacks.Callback):
    ''' Class for visualization of predicted flow '''
    def __init__(self,
                 dataset,
                 target_size=(448, 1024),
                 num_visualize=1,
                 choose_random=False,
                 logdir='predicted_flows',
                 **kwargs):
        '''
        Args:
          dataset (FlowDataset): iterating input images and extra info
            (without GT-flow)
          target_size (tuple of int): size for properly processed in the model.
            RAFT requires size multiple of 64. Default is (448, 1024).
          num_visualize (int): number of images to be saved. Default is 1
          choose_random (bool): whether to shuffle target image. Default is False.
          logdir (str): log directory for resulting visualization. Default is 'logs'
        '''
        super().__init__(**kwargs)
        self.dataset = dataset
        self.target_size = target_size
        self.num_visualize = num_visualize
        self.choose_random = choose_random
        self.logdir = logdir

        if not os.path.exists(logdir):
            os.makedirs(logdir)

    def on_epoch_end(self, epoch, logs=None):
        if self.choose_random:
            vis_ids = np.random.choice(len(self.dataset),
                                       size=self.num_visualize,
                                       replace=False)
        else:
            vis_ids = range(self.num_visualize)
        
        for i in vis_ids:
            image1, image2, (scene, index) = self.dataset[i]
            if len(image1.shape) > 3:
                raise ValueError('target dataset must not be batched')

            h_origin, w_origin, _ = image1.shape

            image1_tf = tf.convert_to_tensor(image1, dtype=tf.float32)
            image2_tf = tf.convert_to_tensor(image2, dtype=tf.float32)

            image1_tf = tf.image.resize_with_crop_or_pad(image1_tf, *self.target_size)
            image2_tf = tf.image.resize_with_crop_or_pad(image2_tf, *self.target_size)

            image1_tf = tf.expand_dims(image1_tf, axis=0)
            image2_tf = tf.expand_dims(image2_tf, axis=0)

            flow_predictions = self.model([image1_tf, image2_tf], training=False)
            flow_pred = flow_predictions[-1][0]
            flow_pred = tf.image.resize_with_crop_or_pad(flow_pred, h_origin, w_origin)
            flow_img = flow_to_image(flow_pred.numpy()).astype(np.uint8)

            contents = np.concatenate([image1, image2, flow_img], axis=0)

            filename = f'epoch{str(epoch+1).zfill(3)}_{scene}_{index}.png'
            savepath = os.path.join(self.logdir, filename)
            imageio.imwrite(savepath, contents)
