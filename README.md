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

# iters means number of recurrent update of flow 
raft = RAFT(iters=iters)
raft.compile(
    optimizer=optimizer,
    loss=sequence_loss,
    epe=end_point_error
)

raft.fit(
    dataset,
    epochs=epochs,
    callbacks=callbacks,
)
```

In practice, you are required to prepare dataset, optimizer, callbacks etc, check details in `train.py`.

## Note
Though I have tried to reproduce the original implementation faithfully, there is some difference between it and my implementation (mainly because of used framework: PyTorch/TensorFlow);

- The original implements cuda-based correlation function but I don't. My TF-based implementation works well, but cuda-based one may runs faster.
- I have trained my model only on MPI-Sintel dataset in my private environment (GCP with P100 accelerator). The model has been trained well, but not reached the best score reported in the paper (trained on multiple datasets).
- The original uses mixed-precision. This may get traininig much faster, but I don't. TensorFlow also enables mixed-precision with few additional lines, see https://www.tensorflow.org/guide/mixed_precision if interested.

## References
- https://github.com/princeton-vl/RAFT
- https://github.com/NVIDIA/flownet2-pytorch
- https://github.com/NVlabs/PWC-Net
