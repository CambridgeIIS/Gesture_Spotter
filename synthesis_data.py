from res_eval import *
from data_utils import *
import numpy as np
import os
import glob
from numpy import genfromtxt
import tensorflow as tf

def synthesis_data(num_synthetic_data,data, path_to_saved_model_g,save_path,max_x_length,max_c_length,input_dim):
    batch_sz = 128
    # date_time = '20210212-173019'
    # path_to_saved_model_g = '/home/shawn/desktop/GAN_DE_gestures/{}/res/model/generator_g_1'.format(date_time)
    # path_to_saved_model_f = '/home/shawn/desktop/GAN_DE_gestures/{}/res/model/generator_f_1'.format(date_time)

    # new_path = '/home/shawn/desktop/KGS/kgs_online_slim/data/logs'
    # path = new_path+'/*.csv'
    print('start loading data ......')
    new_gesture_list = []
    new_label_list = []
    for new_gesture in data:
        new_gesture = interposlate(new_gesture,max_x_length,input_dim)
        new_gesture_list.append(new_gesture)
        new_label_list.append(int(11))
    new_gesture_list = np.array(new_gesture_list, np.float32)
    pseudo_labels = np.ones([batch_sz, max_c_length, 1], np.float32)
    print('finished loading data ......\n')


    print('start loading model ......')
    imported_model_g = tf.saved_model.load(path_to_saved_model_g)
    print('finished loading model ......\n')


    real_input = []
    index = 0
    while len(real_input) < batch_sz:
        index += 1
        if index > len(new_gesture_list):
            index = 0
        real_input.append(new_gesture_list[-1])
    real_input = np.array(real_input, np.float32)
    print('start synthesizing data ......')
    predictions_g = imported_model_g([real_input, pseudo_labels], training=False)
    # predictions_f = imported_model_f([real_input, pseudo_labels], training=False)
    # print('finished synthesizing data ......\n')
    # synthetic_data = []
    # for i in range(num_synthetic_data):
    #     synthetic_data.append(interpolate(predictions_g[i],300,input_dim))

    return synthesis_data, 
    # print('start saving data to {}'.format(save_path+'synthetic_data.npy'))
    # np.save(save_path+'synthetic_data',synthetic_data)
    # print('finished saving data ......')