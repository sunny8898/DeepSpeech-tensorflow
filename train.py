# -*-coding:utf-8-*-
"""
Train an end-to-end speech recognition model using CTC.
Use $python train.py --help for usage
"""

from __future__ import absolute_import, division, print_function
import edit_distance
import argparse
import logging
import os
import numpy
from data_generator import DataGenerator
from model import compile_gru_model, compile_train_fn, compile_test_fn, decode_ctc_fn 
import numpy as np
from utils import configure_logging, save_model, ctc_input_length, argmax_decode
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras as K
logger = logging.getLogger(__name__)

def validation(model, val_fn, decode_fn, datagen, mb_size=64):
    """ Validation routine for speech-models
    Params:
        model (keras.model): Constructed keras model
        val_fn (theano.function): A theano function that calculates the cost
            over a validation set
        datagen (DataGenerator)
        mb_size (int): Size of each minibatch
    Returns:
        val_cost (float): Average validation cost over the whole validation set
    """
    avg_cost = 0.0
    avg_acc = 0.0
    i = 0
    for batch in datagen.iterate_validation(mb_size):
        inputs = batch['x']
        labels = batch['y']
        input_lengths = batch['input_lengths']
        label_lengths = batch['label_lengths']
        texts = batch['texts']
        # print('labels:'+str(labels))
        # Due to convolution, the number of timesteps of the output
        # is different from the input length. Calculate the resulting
        # timesteps
        ctc_input_lens = ctc_input_length(model, input_lengths)
        # print('ctc_input_lens_pre:'+str(ctc_input_lens))
        prediction, ctc_cost = val_fn([inputs, ctc_input_lens, labels,
                              label_lengths, False])
        # print(labels)
        # prediction = np.swapaxes(prediction, 0, 1)
        predict_str = argmax_decode(prediction, decode_fn, ctc_input_lens)
        
        # print('predict_str:'+str(predict_str))
        avg_cost += ctc_cost.mean()
        print('predict_str:'+str(predict_str))
        print('texts:'+str(texts))
        acc_sum = 0
        for index, text in enumerate(texts): 
            sm = edit_distance.SequenceMatcher(a=text,b=predict_str[index])
            acc = 1.0 - sm.distance()/len(text)
            acc_sum = acc_sum + acc
        avg_acc += acc_sum*1.0/(index+1)
        i += 1
    if i == 0:
        return 0.0, 0.0
    return avg_cost / i, avg_acc / i


def train(model, train_fn, val_fn, decode_fn, datagen, save_dir, epochs=10, mb_size=16,
          do_sortagrad=True):
    """ Main training routine for speech-models
    Params:
        model (keras.model): Constructed keras model
        train_fn (theano.function): A theano function that takes in acoustic
            inputs and updates the model
        val_fn (theano.function): A theano function that calculates the cost
            over a validation set
        datagen (DataGenerator)
        save_dir (str): Path where model and costs are saved
        epochs (int): Total epochs to continue training
        mb_size (int): Size of each minibatch
        do_sortagrad (bool): If true, we sort utterances by their length in the
            first epoch
    """
    train_costs, val_costs, val_acc = [], [], []
    iters = 0
    for e in range(epochs):
        #if do_sortagrad:
        #    shuffle = e != 0
        #    sortagrad = e == 0
        #else:
        shuffle = True
        sortagrad = False
        for i, batch in \
                enumerate(datagen.iterate_train(mb_size, shuffle=shuffle,
                                                sort_by_duration=sortagrad)):
            inputs = batch['x']
            labels = batch['y']
            input_lengths = batch['input_lengths']
            label_lengths = np.array(batch['label_lengths'])
            # Due to convolution, the number of timesteps of the output
            # is different from the input length. Calculate the resulting
            # timesteps
            ctc_input_lens = np.array(ctc_input_length(model, input_lengths))
            
            _, ctc_cost = train_fn([inputs, ctc_input_lens, labels,
                                    label_lengths, True])
            train_costs.append(ctc_cost)
            if i % 10 == 0:
                logger.info("Epoch: {}, Iteration: {}, Loss: {}"
                            .format(e, i, ctc_cost.mean(), input_lengths))

            iters += 1
            if iters % 500 == 0:
                val_cost, avg_acc = validation(model, val_fn, decode_fn, datagen, mb_size)
                logger.info("validation,cost:{}".format(val_cost))
                logger.info("validation,acc:{}".format(avg_acc))
                val_costs.append(val_cost)
                val_acc.append(avg_acc)
                save_model(save_dir, model, train_costs, val_costs, val_acc, iters)
        if iters % 500 != 0:
            # End of an epoch. Check validation cost and save costs
            val_cost, avg_acc = validation(model, val_fn, decode_fn, datagen, mb_size)
            logger.info("validation,cost:{}".format(val_cost))
            logger.info("validation,acc:{}".format(avg_acc))
            val_costs.append(val_cost)
            val_acc.append(avg_acc)
            
            # 保存模型
            save_model(save_dir, model, train_costs, val_costs, val_acc, iters)
            

def main(train_desc_file, val_desc_file, epochs, save_dir, sortagrad, NUM_GPU):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Configure logging
    configure_logging(file_log_path=os.path.join(save_dir, 'train_log.txt'))

    # Prepare the data generator
    datagen = DataGenerator()
    # Load the JSON file that contains the dataset
    datagen.load_train_data(train_desc_file)
    datagen.load_validation_data(val_desc_file)
    # Use a few samples from the dataset, to calculate the means and variance
    # of the features, so that we can center our inputs to the network
    feats_mean,feats_std = datagen.fit_train(100)
    np.savetxt(save_dir+'/feats_mean.txt',feats_mean,fmt='%0.8f')
    np.savetxt(save_dir+'/feats_std.txt',feats_std,fmt='%0.8f')
    
    # Compile a Recurrent Network with 1 1D convolution layer, GRU units
    # and 1 fully connected layer
    model = compile_gru_model(recur_layers=3, batch_norm=True, num_gpu=NUM_GPU)
    # 输出模型结构
    logger.info('*'*20)
    model_summary = model.summary()
    logger.info(model_summary)
    logger.info('*'*20)
    # 加载已训练好的模型
    model.load_weights("model_0827_left_30000/model_30000_weights.h5")
    
    # Compile the CTC training function
    train_fn = compile_train_fn(model)
    
    # Compile the validation function
    val_fn = compile_test_fn(model)

    # Compile the CTC decode function
    decode_fn = decode_ctc_fn(model)

    # Train the model
    train(model, train_fn, val_fn, decode_fn, datagen, save_dir, epochs=epochs,
          do_sortagrad=sortagrad)


if __name__ == '__main__':
   
    NUM_GPU = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #进行配置，使用80%的GPU
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    #config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser()
    parser.add_argument('train_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'training labels and paths to the audio files.')
    parser.add_argument('val_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'validation labels and paths to the audio files.')
    parser.add_argument('save_dir', type=str,
                        help='Directory to store the model. This will be '
                             'created if it doesnt already exist')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs to train the model')
    parser.add_argument('--sortagrad', type=bool, default=True,
                        help='If true, we sort utterances by their length in '
                             'the first epoch')
    args = parser.parse_args()

    main(args.train_desc_file, args.val_desc_file, args.epochs, args.save_dir,
         args.sortagrad, NUM_GPU)
