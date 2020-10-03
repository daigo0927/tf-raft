# tf-raft
RAFT (Recurrent All Pairs Field Transforms for Optical Flow, Teed et. al., ECCV2020) implementation via tf.keras

## Original resources
- [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
- https://github.com/princeton-vl/RAFT

## Installation

```
$ pip install tf-raft
```

or you can simply clone this repository.

### Dependencies
- TensorFlow
- TensorFlow-addons
- albumentations

see details in `pyoroject.toml`

## Optical flow datasets
[MPI-Sintel](http://sintel.is.tue.mpg.de/) or [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) datasets are relatively light. See more datasets in the [oirignal repository](https://github.com/princeton-vl/RAFT)

## Usage

``` python
from tf_raft.model import RAFT, SmallRAFT
from tf_raft.losses import sequence_loss, end_point_error

# iters/iters_pred are the number of recurrent update of flow in training/prediction
raft = RAFT(iters=iters, iters_pred=iters_pred)
raft.compile(
    optimizer=optimizer,
    clip_norm=clip_norm,
    loss=sequence_loss,
    epe=end_point_error
)

raft.fit(
    ds_train,
    epochs=epochs,
    callbacks=callbacks,
    steps_per_epoch=train_size//batch_size,
    validation_data=ds_val,
    validation_steps=val_size
)
```

In practice, you are required to prepare dataset, optimizer, callbacks etc, check details in `train_sintel.py` or `train_chairs.py`.

### Train via YAML configuration

`train_chairs.py` and `train_sintel.py` train RAFT model via YAML configuration. Sample configs are in `configs` directory. Run;

``` shell
$ python train_chairs.py /path/to/config.yml
```

## Pre-trained models

I made the pre-trained weights (on both FlyingChairs and MPI-Sintel) public.
You can download them via `gsutil` or `curl`.

### Trained weights on FlyingChairs

``` shell
$ gsutil cp -r gs://tf-raft-pretrained/2020-09-26T18-38/checkpoints .
```
or
``` shell
$ mkdir checkpoints
$ curl -OL https://storage.googleapis.com/tf-raft-pretrained/2020-09-26T18-38/checkpoints/model.data-00000-of-00001
$ curl -OL https://storage.googleapis.com/tf-raft-pretrained/2020-09-26T18-38/checkpoints/model.index
$ mv model* checkpoints/
```

### Trained weights on MPI-Sintel (Clean path)

``` shell
$ gsutil cp -r gs://tf-raft-pretrained/2020-09-26T08-51/checkpoints .
```
or
``` shell
$ mkdir checkpoints
$ curl -OL https://storage.googleapis.com/tf-raft-pretrained/2020-09-26T08-51/checkpoints/model.data-00000-of-00001
$ curl -OL https://storage.googleapis.com/tf-raft-pretrained/2020-09-26T08-51/checkpoints/model.index
$ mv model* checkpoints/
```

### Load weights

``` python
raft = RAFT(iters=iters, iters_pred=iters_pred)
raft.load_weights('checkpoints/model')

# forward (with dummy inputs)
x1 = np.random.uniform(0, 255, (1, 448, 512, 3)).astype(np.float32)
x2 = np.random.uniform(0, 255, (1, 448, 512, 3)).astype(np.float32)
flow_predictions = model([x1, x2], training=False)

print(flow_predictions[-1].shape) # >> (1, 448, 512, 2)
```

## Note
Though I have tried to reproduce the original implementation faithfully, there is some difference between the original one and mine (mainly because of used framework: PyTorch/TensorFlow);

- The original implementations provides cuda-based correlation function but I don't. My TF-based implementation works well, but cuda-based one may run faster.
- I have trained my model on FlyingChairs and MPI-Sintel separately in my private environment (GCP with P100 accelerator). The model has been trained well, but not reached the best score reported in the paper (trained on multiple datasets).
- The original one uses mixed-precision. This may get training much faster, but I don't. TensorFlow also enables mixed-precision with few additional lines, see https://www.tensorflow.org/guide/mixed_precision if interested.

Additional, global gradient clipping seems to be essential for stable training though it is not emphasized in the original paper. This operation can be done via `torch.nn.utils.clip_grad_norm_(model.parameters(), clip)` in PyTorch, `tf.clip_by_global_norm(grads, clip_norm)` in TF (coded at `self.train_step` in `tf_raft/model.py`).

## References
- https://github.com/princeton-vl/RAFT
- https://github.com/NVIDIA/flownet2-pytorch
- https://github.com/NVlabs/PWC-Net
