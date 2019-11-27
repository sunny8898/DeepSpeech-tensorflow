from tensorflow.python.platform import gfile
import tensorflow as tf
from data_generator import DataGenerator
from utils import argmax_decode, conv_output_length, load_model, ctc_input_length
from text import Alphabet
from ds_ctcdecoder import Scorer
import recognize
import os
import time

from memory_profiler import profile

# cpu上跑配置cpu参数
#cpu_num = 1
#config = tf.ConfigProto(device_count={"CPU": 1},
#            inter_op_parallelism_threads = 0,
#            intra_op_parallelism_threads = 0,
#            log_device_placement=True)

# gpu上跑指定gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配

# 加载语言模型并设置权重
lm_binary_path = 'LM_model/lm.klm'
lm_trie_path = 'LM_model/lm_trie'
alphabet_path = 'LM_model/alphabet_zh.txt'
language_model_weight = 0.75
word_count_weight = 1.85
alphabet = Alphabet(os.path.abspath(alphabet_path))
LM_model = Scorer(language_model_weight, word_count_weight, lm_binary_path, lm_trie_path, alphabet)

# 加载声学模型
load_dir = "speech_model/"
speech_model = "speech_model/20190804model.pb"
with gfile.FastGFile(speech_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    input_x,output_y,relu = tf.import_graph_def(graph_def, 
                                                name='', 
                                                return_elements=["acoustic_input:0",
                                                                 "time_distributed_1/Reshape_1:0",
                                                                 "conv1d/Relu:0"]) 
speech_sess = tf.Session(graph=tf.get_default_graph(), config=config)


# 准备数据生成器
datagen = DataGenerator()
# 准备特征标准化参数
datagen.fit_train_test(load_dir)


def trans_compute(audio_list):
    initob = (speech_sess, input_x, output_y, relu, LM_model, alphabet, datagen)
    prediction = recognize.main(audio_list, initob)
    return prediction


def trans_entrance(audio_path):
    text = trans_compute([audio_path])
    return text


if __name__ == '__main__':

    filename = "data_test/13.wav"
    for i in range(1, 5):
        start_time = time.time()
        result = trans_entrance(filename)
        prediction_time = time.time() - start_time
        # print("prediction:{}, time:{}".format(result[0], prediction_time))
        print(prediction_time)
    

