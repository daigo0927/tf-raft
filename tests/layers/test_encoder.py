import pytest
import numpy as np
import tensorflow as tf
import torch

# from tf_raft.layers import encoder
# from RAFT.core import extractor as original_extractor


SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
IMAGE_SIZE = (128, 128)
NUM_CHANNELS = 32
FILTERS = 64
BATCH_SIZE = 16


def convert_weights_tf2torch(tf_weights):
    ndim = len(tf_weights.shape)
    if ndim == 1:
        np_weights = tf_weights.numpy()
        return torch.nn.Parameter(torch.from_numpy(np_weights))
    elif ndim == 2:
        # (in_dim, out_dim) -> (out_dim, in_dim)
        np_weights = tf_weights.numpy()
        return torch.nn.Parameter(torch.from_numpy(np_weights.T))
    elif ndim == 4:
        # (kh, kw, in_ch, out_ch) -> (out_ch, in_ch, kh, kw)
        np_weights = tf_weights.numpy()
        np_weights = np.tranpose(np_weights, (3, 2, 0, 1))
        return torch.nn.Parameter(torch.from_numpy(np_weights))


@pytest.mark.parametrize('norm_type', ['group', 'batch', 'instance', None])
def test_normalization(norm_type):
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, *IMAGE_SIZE, NUM_CHANNELS))
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    inputs_pt = torch.from_numpy(np.transpose(inputs, (0, 3, 1, 2))).float()
    
    groups = 8
    norm = encoder.Normalization(norm_type, groups)
    outputs_tf = norm(inputs_tf, training=True)

    if norm_type == 'group':
        norm_pt = torch.nn.GroupNorm(groups, NUM_CHANNELS, eps=1e-3)
    elif norm_type == 'batch':
        norm_pt = torch.nn.BatchNorm2d(NUM_CHANNELS, eps=1e-3)
    elif norm_type == 'instance':
        norm_pt = torch.nn.InstanceNorm2d(NUM_CHANNELS, eps=1e-3)
    elif norm_type is None:
        norm_pt = torch.nn.Sequential()

    if hasattr(norm_pt, 'weight'):
        norm_pt.weight = convert_weights_tf2torch(norm.gamma)
    if hasattr(norm_pt, 'bias'):
        norm_pt.bias = convert_weights_tf2torch(norm.beta)

    norm_pt.train()
    outputs_pt = norm_pt(inputs_pt)
    outputs_valid = np.transpose(outputs_pt.detach().numpy(), (0, 2, 3, 1))

    np.testing.assert_allclose(outputs_tf.numpy(), outputs_valid, atol=1e-5, rtol=1e-4)


