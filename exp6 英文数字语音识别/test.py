import os
import librosa
import numpy as np
from asrcnn import mean_normalize
import tensorflow as tf


# 准备测试数据

def read_test_wave(path):
    files = os.listdir(path)
    feature = []
    features = []
    label = []
    for wav in files:
        # print(wav)
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])
        wave, sr = librosa.load(path+wav, mono=True)
        label.append(ans)
        # print("真实lable: %d" % ans)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
        feature.append(np.array(mfcc))
    features = mean_normalize(np.array(feature))
    return features,label


def test(path):
    features, label = read_test_wave(path)
    print('loading ASRCNN model...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('cnn_model/model.ckpt-999.meta')
        saver.restore(sess, tf.train.latest_checkpoint('cnn_model'))
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        pred = graph.get_tensor_by_name("pred:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(0, len(label)):
            feed_dict = {input_x: features[i].reshape(1, 20, 100), keep_prob: 1.0}
            test_output = sess.run(pred, feed_dict=feed_dict)

            print("=" * 15)
            print("真实lable: %d" % label[i])
            print("识别结果为:" + str(test_output[0]))
        print("祝贺你!")





path = os.getcwd()+ '\\numbersRec\\test\\'
test(path)