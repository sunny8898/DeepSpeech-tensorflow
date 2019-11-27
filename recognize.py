import tensorflow as tf
from ds_ctcdecoder import ctc_beam_search_decoder_batch

from memory_profiler import profile
@profile(precision=4,stream=open('memory_test.log','w+'))
def test(audio_list, speech_sess, input_x, output_y, relu, LM_model, alphabet, datagen):
    # 特征准备
    batch = datagen.prepare_audio(audio_list[0])
    inputs = batch['x']
    # 调用声学模型
    prediction,relu_result = speech_sess.run([output_y,relu],
                                              feed_dict={input_x:inputs})


    seq_lengths = [relu_result.shape[1]-2]
    # 解码
    predictions = []
    decoded = ctc_beam_search_decoder_batch(prediction, 
                                            seq_lengths, 
                                            alphabet, 
                                            40,   # beam wide 
                                            num_processes=1, 
                                            cutoff_top_n=20, 
                                            scorer=LM_model)
    predictions.extend(d[0][1] for d in decoded)
    return predictions


def main(audio_list, initob):
    # Load the JSON file that contains the dataset
    speech_sess, input_x, output_y, relu, LM_model, alphabet, datagen = initob
    # Test the model
    prediction = test(audio_list, speech_sess, input_x, output_y, relu, LM_model, alphabet, datagen)
    return prediction
