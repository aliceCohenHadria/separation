from tensorflow import keras
import numpy as np
import os
import librosa.display
import matplotlib.pyplot as plt


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_index, dim, N, batch_size, n_batch,
                 rootDB, winL, hop, sr,
                 log=False,  timeDim=0,
                 name_input='mix.npy', name_target='merge.npy'):
        'Initialization'
        # Train parameters
        self.batch_size = batch_size
        self.n_batch = n_batch
        # Data parameters
        self.dim = dim
        self.timeDim = timeDim
        self.N = N
        self.hop = hop
        self.sr=sr
        self.winL = winL

        # Data info
        self.name_target = name_target
        self.name_input = name_input
        self.data_index = data_index
        self.rootDB = rootDB


        # Control
        self.checked = 0
        self.log = log
        self.size_epoch = self.batch_size*self.n_batch
        self.__check()
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.data_index) / self.batch_size))
        return self.n_batch

    def __check(self):
        if len(self.data_index) < self.size_epoch:
            message = ("You are demanding more information in an epoch",
                       " than data points you have")
            raise Exception(message)
        return

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.data_index[
            self.checked:self.checked+self.size_epoch]
        if self.size_epoch == len(self.indexes):
            self.checked += self.size_epoch
        else:
            # Last chunk of data that might be smalller than the epoch size
            d = self.size_epoch - len(self.indexes)
            self.indexes += self.data_index[0:d]
            self.checked = d

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, *self.dim))
        # Generate data
        name_tmp = 'None'
        i=0
        for (name, pos) in indexes:
            # print(name, pos, cond)
            if name_tmp != name:
                target_cond = {}
                path_tmp = name

                data, target = self.__read_data(path_tmp)

                name_tmp = name

            x, y = self.__prepare_data(data, target, pos)
            # Store input
            X[i, ] = x
            # Store target
            Y[i, ] = y

            i+=1

        r = np.random.permutation(X.shape[0])

        return X[r], Y[r]
        # return [X, C], Y

    def __read_data(self, path_data):
        """read mix and separate voice and compute spectrogram """
        mix,_ = librosa.core.load(os.path.join(self.rootDB, path_data, self.name_input), sr=self.sr)
        voice,_ = librosa.core.load(os.path.join(self.rootDB, path_data, self.name_target), sr=self.sr)

        mixSpec = self.__makeSpec(mix)
        voiceSpec = self.__makeSpec(voice)
        
        return mixSpec, voiceSpec


    def __prepare_data(self, data, target, pos):
        """ Take a slice of size N of the data and targert"""
        d = data[:,pos:pos+self.N,:]
        t = target[:,pos:pos+self.N,:]
        return d, t

    def __makeSpec(self, audio):
        """ Compute complexe spectrogram, and stack it (real and imaginary part) """
        s = librosa.core.stft(audio, n_fft=self.winL*2, hop_length=self.hop,
                            win_length=self.winL)[:self.winL,:]
        real = np.real(s.T)
        cmplx = np.imag(s.T)

        real_complex = np.transpose(np.dstack((real, cmplx)), (2 , 0, 1))
        return real_complex