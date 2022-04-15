from sys import call_tracing
import generator as gen
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from functionalModel import unet_tfc_tdf
from tensorflow.keras import activations
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--winL', type=int, default=1024)
    parser.add_argument('--root', type=str, default="/data/anasynth_nonbp/cohenhadria/")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=100)
    parser.add_argument('--sr', type=int, default=22050)

    args = parser.parse_args()


    name_csv = "dataset.csv"
    name_data = "train"

    root = args.root
    winL = args.winL
    hop = winL // 2
    n_fft = winL

    batch_size = args.batch_size
    step_train = 1
    n_batch = args.n_batch
    sr = args.sr
    N = 128
    
    dim = (2, N, n_fft)



    genTrain =  gen.make_data_gen(name_csv, root, name_data, batch_size, step_train, sr, N, dim, n_batch, winL, hop)
    n_blocks = 7
    input_channels = 2
    internal_channels = 24
    n_internal_layers = 5    
    kernel_size_t = 3
    kernel_size_f = 3
    first_conv_activation = activations.relu
    last_activation = None
    t_down_layers = None
    f_down_layers = None


    #Callback to save models
    pathSave = "/data/anasynth_nonbp/cohenhadria/training_full_musdb"

    checkpoint_path = "/data/anasynth_nonbp/cohenhadria/training_full_musdb_long/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)


    i = Input(shape=[2, None, n_fft])

    o = unet_tfc_tdf(i, n_fft*2, n_blocks, input_channels, internal_channels, n_internal_layers, 
                first_conv_activation, last_activation, t_down_layers, f_down_layers, kernel_size_t, kernel_size_f)

    m = Model(inputs = i, outputs = o)
    m.compile(optimizer=RMSprop(), loss='mean_squared_error')

    m.summary()

    m.fit(genTrain, epochs=50, callbacks=cp_callback)
    m.save(pathSave)