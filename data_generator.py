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

    def load_metadata_from_desc_file(self, desc_file, partition='train',
                                     max_duration=20.0,):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        logger.info('Reading description file: {} for partition: {}'
                    .format(desc_file, partition))
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text_hanzi'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    logger.warn('Error reading line #{}: {}'
                                .format(line_num, json_line))
                    logger.warn(str(e))

        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")

    def load_train_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'train')

    def load_test_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'test')

    def load_validation_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'validation')

    @staticmethod
    def sort_by_duration(durations, audio_paths, texts):
        return zip(*sorted(zip(durations, audio_paths, texts)))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def prepare_minibatch(self, audio_paths, texts):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts),\
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [self.featurize(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
            label = text_to_int_sequence(texts[i])
            y.append(label)
            label_lengths.append([len(label)])
        # Flatten labels to comply with warp-CTC signature
        # y = reduce(lambda i, j: i + j, y)
        y = pad_sequences(y, maxlen=len(max(texts, key=len)), dtype='int32',
                          padding='post', truncating='post', value=-1)

        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
        }

    def iterate(self, audio_paths, texts, minibatch_size,
                max_iters=None):
        if max_iters is not None:
            k_iters = max_iters
        else:
            k_iters = int(np.ceil(len(audio_paths) / minibatch_size))
        logger.info("Iters: {}".format(k_iters))
        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        future = pool.submit(self.prepare_minibatch,
                             audio_paths[:minibatch_size],
                             texts[:minibatch_size])
        start = minibatch_size
        for i in range(k_iters - 1):
            wait([future])
            minibatch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_minibatch,
                                 audio_paths[start: start + minibatch_size],
                                 texts[start: start + minibatch_size])
            yield minibatch
            start += minibatch_size
        # Wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

    def iterate_train(self, minibatch_size=16, sort_by_duration=False,
                      shuffle=True):
        return self.iterate(self.train_audio_paths, self.train_texts, minibatch_size)

    def iterate_test(self, minibatch_size=16):
        return self.iterate(self.test_audio_paths, self.test_texts,
                            minibatch_size)

    def iterate_validation(self, minibatch_size=16):
        return self.iterate(self.val_audio_paths, self.val_texts,
                            minibatch_size)


    #def fetch_mean_std(self):
    #    stdstr = '5.19629756  4.74279672  4.25059099  4.0659799   4.26443688  4.46566871   4.51256201  4.55850335  4.59364372  4.61197461  4.60661734  4.60222465   4.6059234   4.60137294  4.61094177  4.61972799  4.61938055  4.60547888   4.59263675  4.59949521  4.58154996  4.599878    4.60391835  4.58341553   4.56717536  4.56338156  4.56349136  4.55839181  4.56537816  4.56436038   4.55024979  4.54563381  4.5388021   4.51715239  4.4985974   4.47877435   4.45160479  4.43499562  4.42688028  4.41892749  4.40728224  4.39547517   4.37866756  4.36988501  4.35934513  4.34689722  4.34640914  4.34614142   4.330982    4.3286629   4.31994563  4.31001422  4.30721158  4.30350678   4.28609375  4.26192003  4.24934757  4.25474776  4.26054438  4.26232912   4.25124763  4.24927111  4.23959638  4.23039383  4.22368628  4.21429862   4.20286398  4.20200652  4.21107314  4.20329076  4.21133147  4.21475908   4.20905467  4.22173025  4.22772622  4.22295576  4.21303099  4.19527195   4.18385384  4.18768311  4.19889702  4.2082337   4.21106565  4.21375397   4.21880614  4.22831121  4.22205113  4.22882881  4.22565121  4.22479832   4.23866574  4.23243573  4.22783183  4.25197827  4.26276392  4.26472953   4.25893701  4.26529189  4.24901741  4.24824148  4.23630761  4.22899418   4.22753357  4.22928455  4.25794054  4.28141211  4.2784007   4.27249864   4.27597364  4.28393766  4.28774771  4.29631217  4.31626494  4.32818759   4.33219728  4.32105298  4.34158807  4.37060799  4.3979097   4.42230784   4.43640866  4.46716567  4.54173872  4.72509461  4.84926506  4.91614409   4.95467545  4.97990786  4.9805346   4.98215343  4.98752051  4.98286396   4.96058     4.95863458  4.95413626  4.93879649  4.93196089  4.92603926   4.91368245  4.90227682  4.88545405  4.87279577  4.86274191  4.84740816   4.83551019  4.82293215  4.81566375  4.80548588  4.79130935  4.77742211   4.76730012  4.76201658  4.75045248  4.73965521  4.72456252  4.71100518   4.70072792  4.70879184  4.71422149  4.70020507  4.74340091'
     #   std = map(lambda x:float(x), re.split('\s+', stdstr))
     #   meanstr = '-19.67197911 -18.69129153 -19.33316901 -19.08022319 -18.58062115  -17.96059861 -17.41224912 -17.00372466 -16.58171845 -16.2644336  -16.16339051 -16.24207221 -16.40029212 -16.52475224 -16.59846463  -16.61529434 -16.60571475 -16.60944207 -16.59673338 -16.57611625  -16.56845953 -16.62413631 -16.70016192 -16.76697019 -16.84609922  -16.94690304 -17.05529385 -17.15620286 -17.25956193 -17.37393251  -17.46681752 -17.5327386  -17.60941354 -17.70231716 -17.80026775  -17.89499201 -17.95240366 -18.01906004 -18.09381336 -18.14965466  -18.2046011  -18.26241449 -18.30959322 -18.36320217 -18.41054925  -18.45748352 -18.49742624 -18.53527678 -18.59030548 -18.64662845  -18.68623162 -18.73096181 -18.76204137 -18.78984637 -18.82436917  -18.83887086 -18.84174931 -18.86898621 -18.89567047 -18.9149211  -18.92506414 -18.9403628  -18.95995593 -18.98981616 -19.00808537  -19.03803653 -19.07169782 -19.11563178 -19.17129955 -19.21749594  -19.28144135 -19.33806006 -19.39941038 -19.45908182 -19.49737062  -19.55104625 -19.60644932 -19.64161908 -19.67268262 -19.70069106  -19.76423176 -19.81701504 -19.86895314 -19.93386678 -19.97880064  -20.02696794 -20.05062451 -20.07471505 -20.09522515 -20.09819132  -20.11286393 -20.09673693 -20.1087908  -20.13844892 -20.15421396  -20.17306627 -20.17250969 -20.18676188 -20.1594725  -20.13769341  -20.12044897 -20.12051472 -20.11498238 -20.11666787 -20.14581742  -20.1958871  -20.20546199 -20.20713582 -20.21825896 -20.22959631  -20.23192934 -20.23219915 -20.25497047 -20.27053584 -20.28220238  -20.27924684 -20.30951856 -20.35706212 -20.40136023 -20.48413654  -20.64931036 -20.90260666 -21.13144934 -21.36682469 -21.49669244  -21.55834811 -21.61261726 -21.65373874 -21.66819749 -21.69962396  -21.75570774 -21.79105995 -21.80749402 -21.84774131 -21.90036908  -21.93642572 -21.9641441  -22.00259363 -22.03232861 -22.04973681  -22.0691499  -22.09999329 -22.1389541  -22.17209621 -22.22598037  -22.25852409 -22.28568008 -22.3176033  -22.34930302 -22.37821813  -22.40648263 -22.43139781 -22.46176989 -22.4850075  -22.51260368  -22.54642915 -22.5651799  -22.57943951 -22.59651579 -22.63026436  -23.44807968'
      #  mean =  map(lambda x:float(x), re.split('\s+', meanstr))
      #  return mean, std


    #def fit_train(self):
    #    """ Estimate the mean and std of the features from the training set
    #    Params:
    #        k_samples (int): Use this number of samples for estimation
    #    """
    #    mean_std = self.fetch_mean_std()
    #    self.feats_mean = np.array(mean_std[0])#np.mean(feats, axis=0)
    #    self.feats_std = np.array(mean_std[1])#np.std(feats, axis=0)


    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)
        return self.feats_mean, self.feats_std
   

    def fit_train_test(self, model_dir):
        self.feats_mean = np.loadtxt(model_dir+'feats_mean.txt',dtype=np.float32)
        self.feats_std = np.loadtxt(model_dir+'feats_std.txt',dtype=np.float32)
