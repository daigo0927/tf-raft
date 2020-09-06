import pytest
import numpy as np
import tensorflow as tf
from tf_raft.losses.losses import sequence_loss, end_point_error, EndPointError


@pytest.fixture
def data():
    # ground truth flow with shape (3, 3, 2)
    flow_gt = np.array([[[0, 1], [0, 2], [0, 3]],
                        [[0, 4], [0, 5], [0, 6]],
                        [[0, 7], [0, 8], [0, 9]]]) - 0.1 # for floating behavior
    valid = np.array([[True, True, True],
                      [True, True, True],
                      [True, True, False]])
    # add batch axis -> (1, 3, 3, 2), (1, 3, 3)
    flow_gt = flow_gt[None, ...]
    valid = valid[None, ...]

    n_predictions = 6
    # as sequential predictions, the last one is used to calculate metrics
    predictions = [np.zeros_like(flow_gt) for _ in range(n_predictions)]
    
    return (flow_gt, valid), predictions


def test_sequence_loss(data):
    (flow_gt, valid), predictions = data

    gamma = 0.8
    n_predictions = len(predictions)

    loss_valid = 0
    for i, flow_pred in enumerate(predictions):
        w = gamma**(n_predictions - i - 1)
        loss = np.abs(flow_pred - flow_gt)
        loss_valid += w*np.mean(valid[..., None]*loss)

    # # as Tensor
    flow_gt = tf.convert_to_tensor(flow_gt, dtype=tf.float32)
    valid = tf.convert_to_tensor(valid, dtype=tf.bool)

    loss_actual = sequence_loss((flow_gt, valid), predictions, gamma=gamma)

    np.testing.assert_almost_equal(loss_actual.numpy(), loss_valid)
    

def test_end_point_error(data):
    (flow_gt, valid), predictions = data

    # as Tensor
    flow_gt = tf.convert_to_tensor(flow_gt, dtype=tf.float32)
    valid = tf.convert_to_tensor(valid, dtype=tf.bool)

    epe_valid = (np.array([1, 2, 3, 4, 5, 6, 7, 8]) - 0.1)**2
    epe_valid = np.mean(np.sqrt(epe_valid))
    u1_valid = 1 / 8
    u3_valid = 3 / 8
    u5_valid = 5 / 8

    info = end_point_error([flow_gt, valid], predictions)

    # 2 decimal for sqrt precision
    np.testing.assert_almost_equal(info['epe'], epe_valid, decimal=2)
    np.testing.assert_almost_equal(info['u1'], u1_valid, decimal=2)
    np.testing.assert_almost_equal(info['u3'], u3_valid, decimal=2)
    np.testing.assert_almost_equal(info['u5'], u5_valid, decimal=2)
