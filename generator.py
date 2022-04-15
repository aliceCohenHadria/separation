import pandas as pd
import random
import itertools
import numpy as np
import os
import glob
import tensorflow as tf
from dataGenerator import DataGenerator
import librosa
import librosa.display
import matplotlib.pylab as plt
from functionalModel import unet_tfc_tdf
from tensorflow.keras import activations
from tensorflow.keras.layers import  Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from argparse import ArgumentParser


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    lres = []
    for i in range(0, len(l), n):
        lres.append(l[i:i + n])
    return lres

def getIndex(db, N, rootDB, step=1, chunk_size=32, lenAudio=10, winL=1024, hop=256,
             sr=8192):
    """ Get all indexes of all examples, moving from step to step"""
    indexes = []
    for ind, entry in db.iterrows():
        n = os.path.join(rootDB, entry['filename'])
        sizeSpec = int((lenAudio * sr - winL) / hop) + 2
        allInd = np.arange(0, sizeSpec-N, step)
        random.shuffle(allInd)
        s=[(n,j) for j in allInd]
        indexes[0:0] = s
    random.shuffle(indexes)
    indexes = chunks(indexes, chunk_size)
    return list(itertools.chain.from_iterable(indexes))

def make_data_gen(name_csv, root, name_data, batch_size, step_train, sr, N, dim, n_batch, winL, hop):
    """create all Index and give them to the DataGenerator Class to feed the model """

    rootDB = os.path.join(root, name_data)
    path_desc = os.path.join(root, name_csv)

    db = pd.read_csv(path_desc, engine='python')
    train_index = getIndex(db, N, rootDB, chunk_size=batch_size, step=step_train, sr=sr, winL=winL, hop=hop)


    g_train = DataGenerator(data_index=train_index, dim=dim, N=N,
                        batch_size=batch_size, n_batch=n_batch,
                        rootDB=rootDB, log=False, winL=winL, hop=hop, sr=sr,
                        name_input="mix.wav", name_target="vocals.wav")
    return g_train



#desc = "/home/cohenhadria/2_recherche/separation/musdb18_extracted/dataset.csv"
#rootDB = "/home/cohenhadria/2_recherche/separation/musdb18_extracted/"
'''root = "/data/anasynth_nonbp/cohenhadria/"
desc = os.path.join(root, "dataset.csv")
rootDB = os.path.join(root, "train")


db = pd.read_csv(desc, engine='python')
step_train = 1

batch_size = 2
n_batch = 2
N = 128
sr = 22050


winL = 1024
hop = winL//2

n_fft = winL//2

dim = (2, N, n_fft)

train_index = getIndex(db, N, rootDB, chunk_size=batch_size, step=step_train, sr=sr)

print("Nb examples " + str(len(train_index)))

g_train = DataGenerator(data_index=train_index, dim=dim, N=N,
                        batch_size=batch_size, n_batch=n_batch,
                        rootDB=rootDB, log=False, winL=winL, hop=hop, sr=sr,
                        name_input="mix.wav", name_target="vocals.wav")
#spec_mix, spec_voice = g_train.__getitem__(0)
#librosa.display.specshow(librosa.amplitude_to_db(spec_mix[2,0,:,:]))
#librosa.display.specshow(librosa.amplitude_to_db(spec_voice[2,0,:,:])

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

checkpoint_path = "training_full_musdb/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


i = Input(shape=[2, None, n_fft])

o = unet_tfc_tdf(i, n_fft*2, n_blocks, input_channels, internal_channels, n_internal_layers, 
            first_conv_activation, last_activation, t_down_layers, f_down_layers, kernel_size_t, kernel_size_f)

m = Model(inputs = i, outputs = o)
m.compile(optimizer=RMSprop(), loss='mean_square_error')

m.summary()

m.fit(g_train, epochs=10)
m.save(pathSave)
# m.save("/home/cohenhadria/2_recherche/separation/model")
'''