from unicodedata import name
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Input, Concatenate, Conv2DTranspose
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

import numpy as np
from define_blocks import TDF, TFC, apply_block

class TFC_TDF_NET(keras.Model):

    def __init__(self,
                 n_fft,
                 n_blocks, input_channels, internal_channels, n_internal_layers,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 kernel_size_t, kernel_size_f):

        def mk_tfc_tdf(internal_channels, f):
            return TFC_TDF(n_internal_layers, internal_channels, kernel_size_t, kernel_size_f, f)

        def mk_tfc_tdf_ds(internal_channels, i, f, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in t_down_layers else (1, 2)
            ds = Sequential()
            ds.add(Conv2D(internal_channels, scale, strides=scale))
            ds.add(BatchNormalization(axis=1))
            return ds, f // scale[-1]

        def mk_tfc_tdf_us(internal_channels, i, f, n, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in [
                    n - 1 - s for s in t_down_layers] else (1, 2)

            us = Sequential()
            us.add(Conv2DTranspose(internal_channels, scale, strides=scale))
            us.add(BatchNormalization(axis=1))
            return us, f * scale[-1]

        super(TFC_TDF_NET, self).__init__()
        assert n_blocks % 2 == 1

        ###########################################################
        # Block-independent Section

        dim_f = n_fft // 2

        self.firstconv_block = [Conv2D(internal_channels, (1, 2), strides=1),
                                BatchNormalization(axis=1),
                                Activation(first_conv_activation)]

        self.encoders = []
        self.downsamplings = []
        self.decoders = []
        self.upsamplings = []
    
        self.lastconv_block = [Conv2D(input_channels, (1, 2), strides=1),  # missing padding(0,1)
                                Activation(last_activation)]
        self.n = n_blocks // 2

        if t_down_layers is None:
            t_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            t_down_layers = list(range(self.n))
        else:
            t_down_layers = string_to_list(t_down_layers)

        if f_down_layers is None:
            f_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            f_down_layers = list(range(self.n))
        else:
            f_down_layers = string_to_list(f_down_layers)


        # Block-independent Section
        ###########################################################

        ###########################################################
        # Block-dependent Section
        f = dim_f

        i = 0
        for i in range(self.n):
            self.encoders.append(mk_tfc_tdf(internal_channels, f))
            ds_layer, f = mk_tfc_tdf_ds(internal_channels, i, f, t_down_layers, name="ds"+i)
            self.downsamplings.append(ds_layer)

        self.mid_block = mk_tfc_tdf(internal_channels, f)

        for i in range(self.n):
            us_layer, f = mk_tfc_tdf_us(internal_channels, i, f, self.n, t_down_layers, name="us"+i)
            self.upsamplings.append(us_layer)
            self.decoders.append(mk_tfc_tdf( internal_channels, f))

        # Block-dependent Section
        ########################################################### 

    def call(self, input):

        x = apply_block(self.firstconv_block, input)
        encoding_outputs = []

        for i in range(self.n):
            x = self.encoders[i](x)
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)

        x = self.mid_block(x)
  
        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = tf.concat([x, encoding_outputs[-i - 1]], 1)
            x = self.decoders[i](x)

        return apply_block(self.lastconv_block, x)





if __name__ == "__main__":
    f = 128



    n_fft = 4096
    n_blocks = 7
    input_channels = 2
    internal_channels = 24
    n_internal_layers = 5    
    kernel_size_t = 3
    kernel_size_f = 3
    first_conv_activation = activations.relu
    last_activation = activations.relu
    t_down_layers = None
    f_down_layers = None

    m = TFC_TDF_NET(n_fft,
                 n_blocks, input_channels, internal_channels, n_internal_layers,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 kernel_size_t, kernel_size_f)

    # a = TFC(5, 3, 2, 2, name="tfc1")
    # b = TDF(24,128, name="tdf2")

    # c = Sequential([Input(shape=[ 2, None, f]), a, b])

