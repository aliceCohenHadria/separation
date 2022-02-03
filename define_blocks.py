from unicodedata import name
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Input, Concatenate
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

import numpy as np


def apply_block(layers, inputs):
    output = inputs
    for layer in layers:
        output = layer(output)
    return output


class TDF(keras.Model):
    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16,
                 activation=activations.relu, name="tdf",):
        super(TDF, self).__init__(name=name)

        if bn_factor is None or bn_factor == "None" or bn_factor == "none":
            self.tdf = [Dense(f, use_bias=bias),
                        BatchNormalization(axis=1),
                        # original code : nn.BatchNorm2d(channels)
                        # It takes input as num_features which is equal
                        # to the number of out-channels of the layer above it.
                        Activation(activation)
                        ]

        else:
            # bottleneck units
            self.bn_units = max(f // bn_factor, min_bn_units)
            self.tdf = [Dense(self.bn_units, use_bias=bias),
                        BatchNormalization(axis=1),
                        Activation(activation),
                        Dense(f, use_bias=bias),
                        BatchNormalization(axis=1),
                        Activation(activation)
                        ]

    def call(self, inputs):
        output = inputs
        for layer in self.tdf:
            output = layer(output)
        return output


class TFC(keras.Model):

    def __init__(self, num_layer, gr, kf, kt, activation=activations.relu, name="tfc"):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__(name=name)
        self.H = []
        for i in range(num_layer):
            self.H.append(
                [Conv2D(gr, (kf, kt), strides=1, padding="same"),
                 BatchNormalization(axis=1),
                 Activation(activation)
                 ]
            )
        self.activation = self.H[-1][-1]



    def call(self, inputs):
        output = inputs
        output_ = apply_block(self.H[0], inputs)
        for layers in self.H[1:]:
            output = tf.concat([output_, output], axis=1)
            output_ = apply_block(layers, output)
        return output_

class TFC_TDF(keras.Model):
    def __init__(self, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=activations.relu, tic_init_mode=None):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins
        below are params for TIF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TFC_TDF, self).__init__()
        self.tfc = TFC(num_layers, gr, kt, kf)
        self.call = TDF(gr, f, bn_factor, bias, min_bn_units)


    def call(self, x):
        x = self.tfc.call(x)
        return x + self.tdf.call(x)


if __name__ == "__main__":
    f = 128

    a = TFC_TDF(3, 3, 2, 2, f)

    c = Sequential([Input(shape=[2, None, f]), a])
    c.summary()

    c.compile()

    i = np.random.rand(2, 2, 34, 128).astype(np.float32)
    o = c.predict(i)
    print(o.shape)
