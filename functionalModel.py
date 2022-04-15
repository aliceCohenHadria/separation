from unicodedata import name
from xml.sax.xmlreader import InputSource
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Input, Concatenate, Conv2DTranspose
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential, Model

import numpy as np



def tdf (input, f, bn_factor=16, bias=False, min_bn_units=16, activation=activations.relu):
    """ tdf block
     channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
    """
    output = input
    if bn_factor is None or bn_factor == "None" or bn_factor == "none":
        output = Dense(f, use_bias=bias) (output)
        output = BatchNormalization(axis=1) (output)
        output = Activation(activation) (output)
    else : 
        bn_units = max(f // bn_factor, min_bn_units) 
        output =  Dense(bn_units, use_bias=bias) (output)
        # print("tdf input dense:", output.shape )

        output = BatchNormalization(axis=1) (output)
        # print("tdf input BatchNormalization:", output.shape )

        output = Activation(activation) (output)

        output = Dense(f, use_bias=bias) (output)
        # print("tdf input dens2:", output.shape )

        output = BatchNormalization(axis=1) (output)
        # print("tdf input BatchNormalization:", output.shape )

        output = Activation(activation) (output)
    return output


def tfc (input, num_layer, gr, kf, kt, activation=activations.relu):
    """in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        activation: activation function"""
    output = input
    #print("tdf input intput:", output.shape )

    output_ = input
    output_ = Conv2D(gr, (kf, kt), strides=1, padding="same") (output_)
    #print("tdf input Conv2D:", output_.shape )

    output_ = BatchNormalization(axis=1) (output_)
    output_ = Activation(activation) (output_)
    

    for i in range(num_layer):
        #print("tfc "+str(i))
        output = Concatenate(axis=1)([output, output_])
        output_ = Conv2D(gr, (kf, kt), strides=1, padding="same") (output)
        output_ = BatchNormalization(axis=1) (output_)
        output_ = Activation(activation) (output_)
    
    return output_


def tfc_tdf (input, num_layer, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=activations.relu):
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

    print("before tfc tdf : ", input.shape)

    output = tfc(input, num_layer, gr, kf, kt, activation)
    a = tdf(output, f, bn_factor, min_bn_units, bias, activation)
    print("tfc tdf : tfc", output.shape, "tdf ", a.shape)
    output = output + tdf(output, f)
    return output


def apply_block(layers, inputs):
    output = inputs
    for layer in layers:
        output = layer(output)
    return output



def mk_tfc_tdf_ds(input, internal_channels, i, f, t_down_layers, name="ds"):
    """ down sampling after tfc tdf block """
    if t_down_layers is None:
        scale = (2, 2)
    else:
        scale = (2, 2) if i in t_down_layers else (1, 2)

    output = input
    output = Conv2D(internal_channels, scale, strides=scale, name=name) (output)
    output = BatchNormalization(axis=1) (output)

    return output, f // scale[-1]

def mk_tfc_tdf_us(input, internal_channels, i, f, n, t_down_layers, name="us"):
    """ upsmapling for the decoding part """
    if t_down_layers is None:
        scale = (2, 2)
    else:
        scale = (2, 2) if i in [
            n - 1 - s for s in t_down_layers] else (1, 2)

    output =  input
    output = Conv2DTranspose(internal_channels, scale, strides=scale, name=name) (output)
    output = BatchNormalization(axis=1) (output)

    return output, f * scale[-1]

def unet_tfc_tdf(input, n_fft, n_blocks, input_channels, internal_channels, n_internal_layers, 
                first_conv_activation, last_activation, t_down_layers, f_down_layers, kt, kf):

    """ Construct a Unet with tfc tdf blocks, and downsmapling (resp upsampling) for encoding 
    (resp decoding) in between each block """
    
    n = n_blocks // 2
    dim_f = n_fft // 2



    if t_down_layers is None:
        t_down_layers = list(range(n))
    elif n_internal_layers == 'None':
        t_down_layers = list(range(n))
    else:
        t_down_layers = string_to_list(t_down_layers)

    if f_down_layers is None:
        f_down_layers = list(range(n))
    elif n_internal_layers == 'None':
        f_down_layers = list(range(n))
    else:
        f_down_layers = string_to_list(f_down_layers)


    output = input


    #first conv block

    output = Conv2D(internal_channels, (1, 2), strides=1, padding="same") (output)
    output = BatchNormalization(axis=1) (output)
    output = Activation(first_conv_activation) (output)

    

    f = dim_f


    encoding_output = []

    i = 0
    for i in range(n):
        print("**************************")
        print("encoding n°" + str(i))
        output = tfc_tdf(output, n_internal_layers, internal_channels, kt, kf, f) 
        encoding_output.append(output)
        output, f = mk_tfc_tdf_ds(output, internal_channels, i, f, t_down_layers, name="ds"+str(i))
        print("after ds ", output.shape)

    print("**************************")
    print("recap encoding")
    for e in encoding_output:
        print(e.shape)
  

    print("**************************")
    print("mid block ")

    output = tfc_tdf(output, n_internal_layers, internal_channels, kt, kf, f)



    for i in range(n):
        print("**************************")
        print("decoding n°" + str(i))

        output, f = mk_tfc_tdf_us(output, internal_channels, i, f, n, t_down_layers, name="us"+str(i))
        print("us shape", output.shape)
        output = Concatenate(axis=1) ([output, encoding_output[-i-1]])
        print("Concatenate shape", output.shape)

        output = tfc_tdf(output, 2*n_internal_layers, internal_channels, kt, kf, f) 

    #last conv block 

    output = Conv2D(input_channels, (1, 2), strides=1, padding="same") (output)  # missing padding(0,1) 
    output = Activation(last_activation) (output)
    
    return output



if __name__=="__main__":

    n_fft = 4096
    f = n_fft//2

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

    i = Input(shape=[2, None, f])

    o = unet_tfc_tdf(i, n_fft, n_blocks, input_channels, internal_channels, n_internal_layers, 
                first_conv_activation, last_activation, t_down_layers, f_down_layers, kernel_size_t, kernel_size_f)

    m = Model(inputs = i, outputs = o)

    nTrames = 8
    ex = np.random.rand(2, 2, nTrames, f).astype(np.float32)
    out_ex = m.predict(ex)
    print(out_ex.shape)