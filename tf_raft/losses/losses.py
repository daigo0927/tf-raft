import tensorflow as tf


def sequence_loss(y_true, y_pred, gamma=0.8, max_flow=400):
    flow_gt, valid = y_true

    n_predictions = len(y_pred)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = tf.sqrt(tf.reduce_sum(flow_gt**2, axis=-1))
    valid = valid & (mag < max_flow)
    # as float and expand channel axis
    valid = tf.expand_dims(tf.cast(valid, tf.float32), axis=-1)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = tf.abs(y_pred[i] - flow_gt)
        flow_loss += i_weight * tf.reduce_mean(valid * i_loss)

    return flow_loss


def end_point_error(y_true, y_pred, max_flow=400):
    flow_gt, valid = y_true
        
    # exclude invalid pixels and extremely large displacements
    mag = tf.sqrt(tf.reduce_sum(flow_gt**2, axis=-1))
    valid = valid & (mag < max_flow)

    epe = tf.sqrt(tf.reduce_sum((y_pred[-1] - flow_gt)**2, axis=-1))
    epe = epe[valid]
    epe_under1 = tf.cast(epe < 1, dtype=tf.float32)
    epe_under3 = tf.cast(epe < 3, dtype=tf.float32)
    epe_under5 = tf.cast(epe < 5, dtype=tf.float32)

    result = {
        'epe': tf.reduce_mean(epe),
        'u1': tf.reduce_mean(epe_under1),
        'u3': tf.reduce_mean(epe_under3),
        'u5': tf.reduce_mean(epe_under5)
    }
    return result


class EndPointError(tf.keras.metrics.Metric):
    ''' Calculates end-point-error and relating metrics '''
    def __init__(self, max_flow=400, **kwargs):
        super().__init__(**kwargs)
        self.max_flow = max_flow

        self.epe = self.add_weight(name='epe', initializer='zeros')
        self.u1 = self.add_weight(name='u1', initializer='zeros')
        self.u3 = self.add_weight(name='u3', initializer='zeros')
        self.u5 = self.add_weight(name='u5', initializer='zeros')

        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        flow_gt, valid = y_true
        
        # exclude invalid pixels and extremely large displacements
        mag = tf.sqrt(tf.reduce_sum(flow_gt**2, axis=-1))
        valid = valid & (mag < self.max_flow)

        epe = tf.sqrt(tf.reduce_sum((y_pred[-1] - flow_gt)**2, axis=-1))
        epe = epe[valid]
        rate_under1 = tf.cast(epe < 1, dtype=tf.float32)
        rate_under3 = tf.cast(epe < 3, dtype=tf.float32)
        rate_under5 = tf.cast(epe < 5, dtype=tf.float32)

        self.epe.assign_add(tf.reduce_mean(epe))
        self.u1.assign_add(tf.reduce_mean(rate_under1))
        self.u3.assign_add(tf.reduce_mean(rate_under3))
        self.u5.assign_add(tf.reduce_mean(rate_under5))
        self.count.assign_add(1)

    def result(self):
        count = tf.cast(self.count, dtype=self.epe.dtype)
        result = {
            'epe': self.epe / count,
            'u1': self.u1 / count,
            'u3': self.u3 / count,
            'u5': self.u5 / count
        }
        return result
