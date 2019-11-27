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
from multi_gpu import ParallelModel


def decode_ctc_fn(model):

    network_output = model.outputs[0]
    ctc_input_lengths = K.placeholder(ndim=1, dtype='int32')

    ctc_decode = K.ctc_decode(network_output, ctc_input_lengths, greedy=True)
    decode_fn = K.function([network_output, ctc_input_lengths], [ctc_decode[0][0]])

    return decode_fn


def compile_train_fn(model, learning_rate=1e-5):
    """ Build the CTC training routine for speech models.
    Args:
        model: A keras model (built=True) instance
    Returns:
        train_fn (theano.function): Function that takes in acoustic inputs,
            and updates the model. Returns network outputs and ctc cost
    """
    logger.info("Building train_fn")
    acoustic_input = model.inputs[0]
    network_output = model.outputs[0]
    label = K.placeholder(ndim=2, dtype='int32')
    label_lens = K.placeholder(ndim=2, dtype='int32')
    ctc_input_lengths = K.placeholder(ndim=2, dtype='int32')

    # ctc_cost = K.mean(K.ctc_batch_cost(label, network_output, ctc_input_lengths, label_lens))
    ctc_cost = K.ctc_batch_cost(label, network_output, ctc_input_lengths, label_lens)
    
    trainable_vars = model.trainable_weights

    optimz = Adam(lr = 1e-4, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)    
    # optimz = SGD(lr=1e-03, clipnorm=100, decay=1e-6, momentum=0.9, nesterov=True)
    updates = list(optimz.get_updates(trainable_vars, [], ctc_cost))
    train_fn = K.function([acoustic_input, ctc_input_lengths, label, label_lens,
                           K.learning_phase()],
                          [network_output, ctc_cost],
                          updates)
    return train_fn


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
    label = K.placeholder(ndim=2, dtype='int32')
    label_lens = K.placeholder(ndim=2, dtype='int32')
    ctc_input_lengths = K.placeholder(ndim=2, dtype='int32')

    # ctc_cost = K.mean(K.ctc_batch_cost(label, network_output, ctc_input_lengths, label_lens))
    ctc_cost = K.ctc_batch_cost(label, network_output, ctc_input_lengths, label_lens)

    val_fn = K.function([acoustic_input, ctc_input_lengths, label, label_lens,
                        K.learning_phase()],
                        [network_output, ctc_cost])
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
        output = normalization.BatchNormalization(name='bn_conv_1d')(conv_1d, training=False)
        # output = normalization.BatchNormalization(name='bn_conv_1d')(conv_1d)
    else:
        output = conv_1d

    for r in range(recur_layers):
        # output = GRU(nodes, activation='relu',
        #              name='rnn_{}'.format(r + 1), init=initialization,
        #              return_sequences=True)(output)
        output = Bidirectional(GRU(nodes, return_sequences=True),name='bi_lstm_{}'.format(r + 1))(output)
        if batch_norm:
            bn_layer = normalization.BatchNormalization(name='bn_rnn_{}'.format(r + 1))
            output = bn_layer(output, training=False)
            #output = bn_layer(output)

    network_output = TimeDistributed(Dense(
        output_dim+1, name='dense', activation='softmax', init=initialization,
    ))(output)
    model = Model(input=acoustic_input, output=network_output)
    #model.conv_output_length = lambda x: conv_output_length(
    #              x, conv_context, conv_border_mode, conv_stride)
    # model = ParallelModel(model, num_gpu)
    return model
