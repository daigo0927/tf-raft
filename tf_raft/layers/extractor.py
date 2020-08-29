import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers


def Normalization(norm_type, groups=None):
    if norm_type == 'group':
        return tfa.layers.GroupNormalization(groups)
    elif norm_type == 'batch':
        return layers.BatchNormalization()
    elif norm_type == 'instance':
        return tfa.layers.InstanceNormalization()
    elif norm_type is None:
        return layers.Lambda(lambda x: x)
    else:
        raise ValueError(f'Invalid norm_type specified: {norm_type}')


class ResBlock(layers.Layer):
    def __init__(self, filters, norm_type='group', strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_type = norm_type
        self.strides = strides

        self.conv1 = layers.Conv2D(filters, 3, strides, 'same')
        self.conv2 = layers.Conv2D(filters, 3, 1, 'same')

        groups = filters // 8
        self.norm1 = Normalization(norm_type, groups)
        self.norm2 = Normalization(norm_type, groups)

        if strides == 1:
            self.downsample = None
        else:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(filters, 1, strides),
                Normalization(norm_type, groups)
            ])

    def call(self, inputs, training):
        fx = inputs
        fx = tf.nn.relu(self.norm1(self.conv1(fx), training=training))
        fx = tf.nn.relu(self.norm2(self.conv2(fx), training=training))

        if self.downsample:
            inputs = self.downsample(inputs, training=training)

        return tf.nn.relu(inputs + fx)


class BottleneckBlock(layers.Layer):
    def __init__(self, filters, norm_type='group', strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_type = norm_type
        self.strides = strides

        self.conv1 = layers.Conv2D(filters//4, 1)
        self.conv2 = layers.Conv2D(filters//4, 3, strides, 'same')
        self.conv3 = layers.Conv2D(filters, 1)

        groups = filters // 8
        self.norm1 = Normalization(norm_type, groups)
        self.norm2 = Normalization(norm_type, groups)
        self.norm3 = Normalization(norm_type, groups)

        if strides == 1:
            self.downsample = None
        else:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(filters, 1, strides),
                Normalization(norm_type, groups)
            ])

    def call(self, inputs, training):
        fx = inputs
        fx = tf.nn.relu(self.norm1(self.conv1(fx), training=training))
        fx = tf.nn.relu(self.norm2(self.conv2(fx), training=training))
        fx = tf.nn.relu(self.norm3(self.conv3(fx), training=training))

        if self.downsample:
            inputs = self.downsample(inputs, training=training)

        return tf.nn.relu(inputs + fx)
        

class BasicEncoder(layers.Layer):
    def __init__(self, output_dim=128, norm_type='batch', drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.drop_rate = drop_rate

        self.conv1 = layers.Conv2D(64, 7, 2, 'same')
        self.norm1 = Normalization(norm_type, groups=8)

        self.layer1 = self._make_layer(64, strides=1)
        self.layer2 = self._make_layer(96, strides=2)
        self.layer3 = self._make_layer(128, strides=2)

        self.conv2 = layers.Conv2D(output_dim, 1)

        self.dropout = layers.Dropout(drop_rate) if drop_rate > 0 else None

    def _make_layer(self, filters, strides):
        seq = tf.keras.Sequential([
            ResBlock(filters, self.norm_type, strides),
            ResBlock(filters, self.norm_type, 1)
        ])
        return seq

    def call(self, inputs, training):
        is_list = isinstance(inputs, (tuple, list))
        if is_list:
            inputs = tf.concat(inputs, axis=0)

        x = tf.nn.relu(self.norm1(self.conv1(inputs), training=training))
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.conv2(x)

        if self.dropout:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, num_or_size_splits=2, axis=0)

        return x


class SmallEncoder(layers.Layer):
    def __init__(self, output_dim=128, norm_type='batch', drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.drop_rate = drop_rate

        self.conv1 = layers.Conv2D(32, 7, 2, 'same')
        self.norm1 = Normalization(norm_type, groups=8)

        self.layer1 = self._make_layer(32, strides=1)
        self.layer2 = self._make_layer(64, strides=2)
        self.layer3 = self._make_layer(96, strides=2)

        self.conv2 = layers.Conv2D(output_dim, 1)

        self.dropout = layers.Dropout(drop_rate) if drop_rate > 0 else None

    def _make_layer(self, filters, strides):
        seq = tf.keras.Sequential([
            ResBlock(filters, self.norm_type, strides),
            ResBlock(filters, self.norm_type, 1)
        ])
        return seq

    def call(self, inputs, training):
        is_list = isinstance(inputs, (tuple, list))
        if is_list:
            inputs = tf.concat(inputs, axis=0)
            
        x = tf.nn.relu(self.norm1(self.conv1(inputs), training=training))
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.conv2(x)

        if self.dropout:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, num_or_size_splits=2, axis=0)

        return x
