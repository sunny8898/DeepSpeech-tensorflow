"""
Define functions used to construct a multilayer GRU CTC model, and
functions for training and testing it.
"""
import sys
import logging
import keras.backend as K
from keras.layers import normalization, Conv1D
from keras.layers import (Convolution1D, Dense, LSTM, Bidirectional,
                          Input, GRU, TimeDistributed)
from keras.models import Model
from keras.optimizers import Adam, SGD
from utils import conv_output_length
import numpy
import keras.regularizers as regularizers
logger = logging.getLogger(__name__)


def compile_test_fn(model):
    """ Build a testing routine for speech models.
    Args:
        model: A keras model (built=True) instance
    Returns:
        val_fn (theano.function): Function that takes in acoustic inputs,
            and calculates the loss. Returns network outputs and ctc cost
    """
    logger.info("Building val_fn")
    acoustic_input = model.inputs[0]
    network_output = model.outputs[0]
    ctc_input_lengths = K.placeholder(ndim=2, dtype='int32')


    val_fn = K.function([acoustic_input, ctc_input_lengths,
                        K.learning_phase()],
                        [network_output])
    return val_fn


def compile_gru_model(input_dim=101, output_dim=4563, recur_layers=3, nodes=1000,
                      conv_context=11, conv_border_mode='valid', conv_stride=2,
                      initialization='glorot_uniform', batch_norm=True, num_gpu=1):
    """ Build a recurrent network (CTC) for speech with GRU units """
    logger.info("Building gru model")
    # Main acoustic input
    acoustic_input = Input(shape=(None, input_dim), name='acoustic_input')

    # Setup the network
    #conv_1d = Conv1D(nodes, conv_context, name='conv_1d',
    #                 padding='same', strides=conv_stride,
    #                 kernel_initializer=initialization,
    #                 activation='relu')(acoustic_input)
    conv_1d = Convolution1D(nodes, conv_context, name='conv1d',
                        border_mode=conv_border_mode,
                        subsample_length=conv_stride, init=initialization,
                        activation='relu')(acoustic_input)
    if batch_norm:
        output = normalization.BatchNormalization(name='bn_conv_1d')(conv_1d, training=True)
    else:
        output = conv_1d

    for r in range(recur_layers):
        # output = GRU(nodes, activation='relu',
        #              name='rnn_{}'.format(r + 1), init=initialization,
        #              return_sequences=True)(output)
        output = Bidirectional(GRU(nodes, return_sequences=True),name='bi_lstm_{}'.format(r + 1))(output)
        if batch_norm:
            bn_layer = normalization.BatchNormalization(name='bn_rnn_{}'.format(r + 1),moving_mean_initializer='zeros')
            output = bn_layer(output, training=True)

    network_output = TimeDistributed(Dense(
        output_dim+1, name='dense', activation='softmax', init=initialization,
    ))(output)
    model = Model(input=acoustic_input, output=network_output)
    #model.conv_output_length = lambda x: conv_output_length(
    #              x, conv_context, conv_border_mode, conv_stride)
    # model = ParallelModel(model, num_gpu)
    return model
