import tensorflow as tf
from tensorflow.keras import layers


class FlowHead(layers.Layer):
    def __init__(self, filters=256, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.conv1 = layers.Conv2D(filters, 3, 1, 'same')
        self.conv2 = layers.Conv2D(2, 3, 1, 'same')

    def call(self, inputs):
        return self.conv2(tf.nn.relu(self.conv1(inputs)))
        

class ConvGRU(layers.Layer):
    def __init__(self, filters=128, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.convz = layers.Conv2D(filters, 3, 1, 'same')
        self.convr = layers.Conv2D(filters, 3, 1, 'same')
        self.convq = layers.Conv2D(filters, 3, 1, 'same')
    
    def call(self, inputs):
        h, x = inputs
        hx = tf.concat([h, x], axis=-1)

        z = tf.nn.sigmoid(self.convz(hx))
        r = tf.nn.sigmoid(self.convr(hx))
        q = tf.nn.tanh(self.convq(tf.concat([r*h, x], axis=-1)))

        h = (1-z)*h + z*q
        return h


class SepConvGRU(layers.Layer):
    def __init__(self, filters=128, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.convz1 = layers.Conv2D(filters, (1, 5), 1, 'same')
        self.convr1 = layers.Conv2D(filters, (1, 5), 1, 'same')
        self.convq1 = layers.Conv2D(filters, (1, 5), 1, 'same')

        self.convz2 = layers.Conv2D(filters, (5, 1), 1, 'same')
        self.convr2 = layers.Conv2D(filters, (5, 1), 1, 'same')
        self.convq2 = layers.Conv2D(filters, (5, 1), 1, 'same')

    def call(self, inputs):
        h, x = inputs
        hx = tf.concat([h, x], axis=-1)
        # horizontal
        z = tf.nn.sigmoid(self.convz1(hx))
        r = tf.nn.sigmoid(self.convr1(hx))
        q = tf.nn.tanh(self.convq1(tf.concat([r*h, x], axis=-1)))
        h = (1-z)*h + z*q

        # vertical
        hx = tf.concat([h, x], axis=-1)
        z = tf.nn.sigmoid(self.convz2(hx))
        r = tf.nn.sigmoid(self.convr2(hx))
        q = tf.nn.tanh(self.convq2(tf.concat([r*h, x], axis=-1)))
        h = (1-z)*h + z*q
        
        return h
    

class SmallMotionEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convc1 = layers.Conv2D(96, 1, 1, 'same')
        self.convf1 = layers.Conv2D(64, 7, 1, 'same')
        self.convf2 = layers.Conv2D(32, 3, 1, 'same')
        self.conv = layers.Conv2D(80, 3, 1, 'same')

    def call(self, inputs):
        flow, corr = inputs
        cor = tf.nn.relu(self.convc1(corr))
        flo = tf.nn.relu(self.convf1(flow))
        flo = tf.nn.relu(self.convf2(flo))
        cor_flo = tf.concat([cor, flo], axis=-1)
        out = tf.nn.relu(self.conv(cor_flo))
        return tf.concat([out, flow], axis=-1)


class BasicMotionEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convc1 = layers.Conv2D(256, 1, 1)
        self.convc2 = layers.Conv2D(192, 3, 1, 'same')
        self.convf1 = layers.Conv2D(128, 7, 1, 'same')
        self.convf2 = layers.Conv2D(64, 3, 1, 'same')
        self.conv = layers.Conv2D(128-2, 3, 1, 'same')

    def call(self, inputs):
        flow, corr = inputs
        cor = tf.nn.relu(self.convc1(corr))
        cor = tf.nn.relu(self.convc2(cor))
        flo = tf.nn.relu(self.convf1(flow))
        flo = tf.nn.relu(self.convf2(flo))

        cor_flo = tf.concat([cor, flo], axis=-1)
        out = tf.nn.relu(self.conv(cor_flo))
        return tf.concat([out, flow], axis=-1)


class SmallUpdateBlock(layers.Layer):
    def __init__(self, filters=96, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.encoder = SmallMotionEncoder()
        self.gru = ConvGRU(filters)
        self.flow_head = FlowHead(128)

    def call(self, inputs):
        net, inp, corr, flow = inputs
        motion_features = self.encoder([flow, corr])
        inp = tf.concat([inp, motion_features], axis=-1)
        net = self.gru([net, inp])
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(layers.Layer):
    def __init__(self, filters=128, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(filters)
        self.flow_head = FlowHead(256)

        self.mask = tf.keras.Sequential([
            layers.Conv2D(256, 3, 1, 'same'),
            layers.ReLU(),
            layers.Conv2D(64*9, 1, 1)
        ])

    def call(self, inputs):
        net, inp, corr, flow = inputs
        motion_features = self.encoder([flow, corr])
        inp = tf.concat([inp, motion_features], axis=-1)

        net = self.gru([net, inp])
        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        mask = 0.25*self.mask(net)
        return net, mask, delta_flow
    
