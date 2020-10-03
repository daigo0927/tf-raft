import numpy as np
import tensorflow as tf

from tf_raft.model import SmallRAFT, RAFT


SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
BATCH_SIZE = 4
IMAGE_SIZE = (64, 96)


def test_upsample_flow():
    # 3x3x2
    flow = np.array([[[1, 2], [3, 4], [5, 6]],
                     [[7, 8], [9, 10], [11, 12]],
                     [[13, 14], [15, 16], [17, 18]]])
    # unfold -> 2x2x8
    flow_unfold = np.array([[[1, 2, 3, 4, 7, 8, 9, 10], [3, 4, 5, 6, 9, 10, 11, 12]],
                            [[7, 8, 9, 10, 13, 14, 15, 16], [9, 10, 11, 12, 15, 16, 17, 18]]])
    # depth2space -> 4x4x2
    up_flow = np.array([[[1, 2], [3, 4], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [9, 10], [11, 12]],
                       [[7, 8], [9, 10], [9, 10], [11, 12]],
                       [[13, 14], [15, 16], [15, 16], [17, 18]]])

    flow_tf = tf.convert_to_tensor(flow, dtype=tf.float32)
    # shape (1, 3, 3, 2)
    flow_tf = tf.reshape(flow_tf, (1, *flow_tf.shape))
    # shape (1, 2, 2, 8)
    flow_unfold_tf = tf.image.extract_patches(flow_tf,
                                              sizes=(1, 2, 2, 1),
                                              strides=(1, 1, 1, 1),
                                              rates=(1, 1, 1, 1),
                                              padding='VALID')
    # shape (1, 4, 4, 2)
    up_flow_tf = tf.nn.depth_to_space(flow_unfold_tf, 2)

    np.testing.assert_allclose(flow_unfold_tf.numpy()[0], flow_unfold)
    np.testing.assert_allclose(up_flow_tf.numpy()[0], up_flow)


def test_raft():
    iters = 6
    iters_pred = 12
    image1 = tf.random.normal((BATCH_SIZE, *IMAGE_SIZE, 3))
    image2 = tf.random.normal((BATCH_SIZE, *IMAGE_SIZE, 3))

    model = RAFT(drop_rate=0.0, iters=iters, iters_pred=iters_pred)
    output = model([image1, image2], training=True)
    assert len(output) == iters
    for flow in output:
        assert flow.shape == (BATCH_SIZE, *IMAGE_SIZE, 2)

    output = model([image1, image2], training=False)
    assert len(output) == iters_pred
    for flow in output:
        assert flow.shape == (BATCH_SIZE, *IMAGE_SIZE, 2)

        
def test_small_raft():
    iters = 6
    iters_pred = 12
    image1 = tf.random.normal((BATCH_SIZE, *IMAGE_SIZE, 3))
    image2 = tf.random.normal((BATCH_SIZE, *IMAGE_SIZE, 3))

    model = SmallRAFT(drop_rate=0.0, iters=iters, iters_pred=iters_pred)
    output = model([image1, image2], training=True)
    assert len(output) == iters
    for flow in output:
        assert flow.shape == (BATCH_SIZE, *IMAGE_SIZE, 2)

    output = model([image1, image2], training=False)
    assert len(output) == iters_pred
    for flow in output:
        assert flow.shape == (BATCH_SIZE, *IMAGE_SIZE, 2)
