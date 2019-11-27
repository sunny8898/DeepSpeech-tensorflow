"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function
from functools import reduce
import os
import json
import logging
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from concurrent.futures import ThreadPoolExecutor, wait
import re
from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence

RNG_SEED = 123
logger = logging.getLogger(__name__)


class DataGenerator(object):
    def __init__(self, step=10, window=25, max_freq=4000, desc_file=None):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq)


    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)


    def prepare_audio(self, audio_paths):
        features = [self.featurize(audio_paths)]
        #features = [self.featurize(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        feat = self.normalize(features[0])  # Center using means and std
        x[0, :feat.shape[0], :] = feat
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'input_lengths': input_lengths  # list(int) Length of each input
        }


    def fit_train_test(self, model_dir):
        self.feats_mean = np.loadtxt(model_dir+'/feats_mean.txt',dtype=np.float32)
        self.feats_std = np.loadtxt(model_dir+'/feats_std.txt',dtype=np.float32)
