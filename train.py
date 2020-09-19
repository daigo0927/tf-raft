import tensorflow as tf
import tensorflow_addons as tfa
import argparse
from functools import partial

from tf_raft.model import RAFT, SmallRAFT
from tf_raft.losses import sequence_loss, end_point_error
from tf_raft.datasets import MpiSintel, FlyingChairs, ShapeSetter, CropOrPadder
from tf_raft.training import VisFlowCallback, first_cycle_scaler


def train(args):
    try:
        sintel_dir = args.sintel_dir
        chairs_dir = args.chairs_dir
        chairs_split_txt = args.chairs_split_txt
        
        epochs = args.epochs
        batch_size = args.batch_size
        iters = args.iters
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay

        crop_size = args.crop_size        
        min_scale = args.min_scale
        max_scale = args.max_scale
        do_flip = args.do_flip

        resume = args.resume
    except ValueError:
        print('invalid arguments are given')
        
    aug_params = {
        'crop_size': crop_size,
        'min_scale': min_scale,
        'max_scale': max_scale,
        'do_flip': do_flip
    }
    
    ds_train_sintel = MpiSintel(aug_params,
                                split='training',
                                root=sintel_dir,
                                dstype='clean')
    ds_train_chairs = FlyingChairs(aug_params,
                                   split='training',
                                   split_txt=chairs_split_txt,
                                   root=chairs_dir)
    ds_train = 20*ds_train_sintel + ds_train_chairs
    train_size = len(ds_train)
    print(f'Found {train_size} samples for training')

    ds_val = MpiSintel(split='training',
                       root=sintel_dir,
                       dstype='clean')
    val_size = len(ds_val)
    print(f'Found {val_size} samples for validation')
    
    ds_test = MpiSintel(split='test',
                        root=sintel_dir,
                        dstype='clean')

    ds_train = tf.data.Dataset.from_generator(
        ds_train,
        output_types=(tf.uint8, tf.uint8, tf.float32, tf.bool),
    )
    train_size = 8
    val_size = 4
    ds_val.image_list = ds_val.image_list[:val_size]
    ds_val.flow_list = ds_val.flow_list[:val_size]
    
    ds_train = ds_train.shuffle(buffer_size=train_size)\
                       .repeat(epochs)\
                       .batch(batch_size)\
                       .map(ShapeSetter(batch_size, crop_size))\
                       .prefetch(buffer_size=1)

    ds_val = tf.data.Dataset.from_generator(
        ds_val,
        output_types=(tf.uint8, tf.uint8, tf.float32, tf.bool),
    )
    ds_val = ds_val.batch(1)\
                   .map(ShapeSetter(batch_size=1, image_size=(436, 1024)))\
                   .map(CropOrPadder(target_size=(448, 1024)))\
                   .prefetch(buffer_size=1)

    scheduler = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=learning_rate,
        maximal_learning_rate=2*learning_rate,
        step_size=1000,
        scale_fn=first_cycle_scaler,
        scale_mode='cycle',
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=weight_decay,
        learning_rate=scheduler
    )

    raft = RAFT(drop_rate=0, iters=iters)
    raft.compile(
        optimizer=optimizer,
        loss=sequence_loss,
        epe=end_point_error
    )

    if resume:
        print('Restoring pretrained weights ...', end=' ')
        raft.load_weights(resume)
        print('done')

    callbacks = [
        tf.keras.callbacks.TensorBoard(),
        VisFlowCallback(ds_test, num_visualize=1, choose_random=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/model',
            save_weights_only=True,
            monitor='val_epe',
            mode='min',
            save_best_only=True
        )
    ]

    raft.fit(
        ds_train,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=train_size//batch_size,
        validation_data=ds_val,
        validation_steps=val_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training RAFT')
    parser.add_argument('-sd', '--sintel_dir', type=str, required=True,
                        help='Path to the MPI-Sintel dataset directory')
    parser.add_argument('-cd', '--chairs_dir', type=str, required=True,
                        help='Path to the FlyingChairs dataset directory')
    parser.add_argument('--chairs-split-txt', type=str, required=True,
                        help='Path to the FlyingChairs split textfile')
    
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of epochs [10]')
    parser.add_argument('-bs', '--batch_size', default=4, type=int,
                        help='Batch size [4]')
    parser.add_argument('-lr', '--learning_rate', default=1.2e-4, type=float,
                        help='Learning rate [1.2e-4]')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay in optimizer [1e-5]')

    parser.add_argument('--iters', default=6, type=int,
                        help='Number of iterations in RAFT inference [6]')

    parser.add_argument('--crop_size', nargs=2, type=int, default=[384, 512],
                        help='Crop size for raw image [384, 512]')
    parser.add_argument('--min_scale', default=-0.1, type=float,
                        help='Minimum scale in augmentation [-0.1]')
    parser.add_argument('--max_scale', default=0.1, type=float,
                        help='Maximum scale in augmentation [0.1]')
    parser.add_argument('--disable_flip', dest='do_flip', action='store_false',
                        help='Disable flip in augmentation [True]')
    parser.set_defaults(do_flip=True)

    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Pretrained checkpoints [None]')
    args = parser.parse_args()

    print('----  Config  ----')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('------------------')

    train(args)
