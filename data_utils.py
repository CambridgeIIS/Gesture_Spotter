import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from model import cnnModel, resNet, Inception, rnn_att_model, smallCnnModel, resNet_LSTM, LSTM_RES
import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from random import randint,shuffle
import numpy as np
import itertools
import logging
import glob
from numpy import genfromtxt

# plt.switch_backend('agg')

import matplotlib.pyplot as plt
import os

def denoise_all(stroke_all, input_dim):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    stroke_new_all = []
    for coords in stroke_all:
        stroke = denoise(coords, input_dim)
        stroke_new_all.append(stroke)
    return stroke_new_all

def interpolate_all(stroke_all, max_x_length, input_dim):
    """
    interpolates strokes using cubic spline
    """
    coords_all = []
    for stroke in stroke_all:
        coords = interpolate(stroke, max_x_length, input_dim)
        coords_all.append(coords)
    return coords_all

def interpolate(stroke, max_x_length, input_dim):
    coords = np.zeros([input_dim, max_x_length], dtype=np.float32)
    if len(stroke) > 3:
        for j in range(input_dim):
            f_x = interp1d(np.arange(len(stroke)), stroke[:, j], kind='linear')
            xx = np.linspace(0, len(stroke) - 1, max_x_length)
            # xx = np.random.uniform(0,len(stroke)-1, max_x_length)
            x_new = f_x(xx)
            coords[j, :] = x_new
    coords = np.transpose(coords)
    return coords


#
def multiplier(data,label,multi):
    data_ = data
    label_ = label
    for i in range(multi):
        data_ = np.concatenate((data_, data))
        label_ = np.concatenate((label_, label))

    data = data_
    label = label_

    return data,label

def denoise(coords, input_dim):
    stroke = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    for i in range(1, input_dim):
        x_new = savgol_filter(coords[:, i], 7, 3, mode='nearest')
        stroke = np.hstack([stroke.reshape(len(coords), -1), x_new.reshape(-1, 1)])
    return stroke


def shuffle(data,label):
    data = np.asarray(data,dtype = object)
    shuffled_indexes = np.random.permutation(np.shape(data)[0])
    data = data[shuffled_indexes]
    label = label[shuffled_indexes]
    return data, label


def relative_track_batch(length, input_dim, data):
    ### do this every iteration to make first step replacement=0
    temp = data[:, 0:input_dim]
    lastplace = np.zeros([length, 1])
    # x
    lastplace[1: length, 0] = temp[0: length - 1, 0]
    lastplace[0, 0] = temp[0, 0]
    temp[0: length, 0] -= lastplace[:, 0]
    # y
    lastplace[1: length, 0] = temp[0: length - 1, 1]
    lastplace[0, 0] = temp[0, 1]
    temp[0: length, 1] -= lastplace[:, 0]
    # z
    lastplace[1: length, 0] = temp[0: length - 1, 2]
    lastplace[0, 0] = temp[0, 2]
    temp[0: length, 2] -= lastplace[:, 0]
    temp = np.reshape(temp, [length, input_dim])
    return temp


def nortowrist(data, xlist):
    ###### normalize to wrist
    ylist = np.add(xlist, 1)
    zlist = np.add(xlist, 2)

    xc = data[:, 0]
    yc = data[:, 1]
    zc = data[:, 2]
    data[:, xlist] -= np.tile(
        np.reshape(xc, [-1, 1]), (1, len(xlist)))
    data[:, ylist] -= np.tile(
        np.reshape(yc, [-1, 1]), (1, len(ylist)))
    data[:, zlist] -= np.tile(
        np.reshape(zc, [-1, 1]), (1, len(zlist)))

    return data


def preprocess(data, xlist,nor_to_wrist=False, relative=False):
    # relative replacement or not
    # use center as track or not, if not, use wrist as track
    length = np.shape(data)[0]
    input_dim = np.shape(data)[1]

    if relative:
        if nor_to_wrist:
            data = nortowrist(data, xlist)
            data = relative_track_batch(length, input_dim, data)
        else:
            data = relative_track_batch(length, input_dim, data)
    elif not relative:
        if nor_to_wrist:
            data = nortowrist(data, xlist)
        else:
            data = data

    return data


def get_new_gesture_data(path,max_length,input_dim, data_synthesis = False):
    new_gesture_list = []

    for i, fname in enumerate(sorted(glob.glob(path))):
        new_gesture = genfromtxt(fname, delimiter=',')
        #     new_gesture = denoise(new_gesture,input_dim)
        new_gesture = interpolate(new_gesture, max_length, input_dim)
        new_gesture_list.append(new_gesture)

    if data_synthesis:
        synthetic_data = np.load('synthetic_data/synthetic_data.npy', allow_pickle=True)
        new_gesture_list = np.concatenate((new_gesture_list, synthetic_data))
    new_label_list = np.ones(len(new_gesture_list)) * 11
    new_gesture_list , new_label_list = shuffle(new_gesture_list,new_label_list)
    return new_gesture_list , new_label_list

def train_test_split(data,label,new_gesture_list,new_label_list,split_num):

    for _ in range(10):
        data, label =  shuffle(data, label)

    train_data, test_data = np.split(
        data, [np.shape(data)[0] * 9 // 10]
    )
    train_label, test_label = np.split(
        label, [np.shape(label)[0] * 9 // 10]
    )
    multi = 10

    train_data = np.concatenate((train_data,new_gesture_list[:split_num]))
    train_label = np.concatenate((train_label,new_label_list[:split_num]))
    train_data, train_label = multiplier(train_data, train_label, multi)

    test_data = np.concatenate((test_data,new_gesture_list[split_num:]))
    test_label = np.concatenate((test_label,new_label_list[split_num:]))
    test_data, test_label = multiplier(test_data, test_label, multi)

    train_data, train_label = shuffle(train_data, train_label)

    test_data, test_label = shuffle(test_data, test_label)

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    train_label = np.asarray(train_label, dtype=np.int32)
    test_label = np.asarray(test_label, dtype=np.int32)

    return train_data, train_label, test_data, test_label

def confusion_matrix(save_path,new_gesture):
    cm = np.load(save_path + '/confusion_matrix.npy')

    classes = ['still', 'ticktock', 'shrink', 'push', 'peaceout', 'madriddles',
               'grow', 'flamingo', 'execution', 'cheshiredance', 'caterpillar', new_gesture]

    target_names = list(classes)
    title = 'Confusion matrix'
    cmap = None
    normalize = True

    accuracy = np.trace(cm) / float(np.sum(cm))
    # np.save(SAVE_PATH+ 'confusion_matrix.npy',cm)
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize=30)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=20)
        plt.yticks(tick_marks, target_names, fontsize=20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 10 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.grid(None)
    plt.tight_layout()
    # plt.ylabel('True label', fontsize=50)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=30)
    # plt.show()
    plt.savefig('confusion_matrix.pdf')
    # plt.close()


def model_selection(Model,drop_out,max_length,input_dim ,output_dim):
    model = resNet_LSTM(max_length,
                          input_dim,
                          output_dim,dropout=drop_out)

    if Model == 'RNN':
        model = rnn_att_model(max_length,
                          input_dim,
                          output_dim,dropout=drop_out)

    if Model == 'SA-ResLSTM':
        model = resNet_LSTM(max_length,
                          input_dim,
                          output_dim,dropout=drop_out)

    elif Model == 'ResNet':
        model = resNet(max_length,
                          input_dim,
                          output_dim)
    elif Model == 'CNN':
        model = cnnModel(max_length,
                          input_dim,
                          output_dim)
    elif Model == 'Inception':
        model = Inception(max_length,
                          input_dim,
                          output_dim)
    elif Model == 'smallCNN':
        model = smallCnnModel(max_length,
                          input_dim,
                          output_dim)

    elif Model == 'LSTM_RES':
        model = LSTM_RES(max_length,
                          input_dim,
                          output_dim,dropout=drop_out)

    return model


def init_logging(log_dir, Model):
    logging_level = logging.INFO

    log_file = 'log_{}.txt'.format(Model)

    log_file = os.path.join(log_dir, log_file)
    if os.path.isfile(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        level=logging_level,
        format='[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    return logging
