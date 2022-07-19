
from fileinput import filename
import matplotlib.pyplot as plt
from res_eval import *
from numpy import genfromtxt
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_accuracy_nttd_(y_pred_list, y_true, y_pred_value_list, threshold=0.6, recur_threshold=20):
    pred_list = []
    # y_pred_value_list_ = (np.array(y_pred_value_list) - min) / (max - min)
    y_pred_value_list_ = y_pred_value_list

    recur = 0
    frame_idx = []
    y_pred_ = y_pred_list[0]
    for i, value in enumerate(y_pred_value_list_):

        if value > threshold:
            recur += 1
            if recur > recur_threshold:
                y_pred = int(y_pred_list[i])
                if y_pred != y_pred_:
                    if y_pred != 0:
                        y_pred_ = y_pred
                        frame_idx.append(i)
                        frame_idx.append(i + 1)
                        pred_list.append(y_pred)
                        recur = 0

    y_pred = np.asarray(pred_list, dtype=np.int64)

    y_true = np.asarray(y_true, dtype=np.int64)

    accuracy = 1 - editdistance.eval(y_pred.tostring(), y_true.tostring()) / len(y_true.tostring())

    print('Pred {}'.format(y_pred))
    print('True {}'.format(y_true))
    print('Accuracy {}'.format(accuracy))

    return y_pred, y_true, accuracy, frame_idx

def generate_online_plot(gesture_classes,filename, save_path,max_length,window_step,num_class=7):


    data = genfromtxt(filename, delimiter=',')


    model = load_model(save_path+'/')

    y_pred_list, outputs_list, y_pred_value_list = online_rec(model, max_length, window_step, data)


    # def online_plot(outputs_list, input_test,input_label, frame_sequence):

    outputs_list_ = np.squeeze(outputs_list, axis=1)
    color_list = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-',
                    'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--',
                    'b-.', 'g-.', 'r-.', 'c-.', 'm-.', 'y-.', 'k-.',
                    'b:', 'g:', 'r:', 'c:', 'm:', 'y:', 'k:', 'b.']


    fig, ax = plt.subplots(figsize=(10, 10), sharex=True)
    # plt.figure(figsize=(10, 10))
    for i in range(num_class):
        # print(outputs_list_[:, i])
        ax.plot(outputs_list_[:, i], label=gesture_classes[i])
    ax.legend()
    plt.show()
    # fig.savefig(save_path+'/online.png', dpi=fig.dpi)
    return y_pred_list, outputs_list, y_pred_value_list



    # return  y_pred_list, y_pred_value_list


if __name__ == "__main__":
    filename = 'test_1/processed_data/1.csv'
    save_path = 'test_1/'
    max_length = 20
    window_step = 5
    num_class = 7


    threshold=0.6
    recur_threshold=20
    max = 0.3
    min = 0.08

    y_true = [1,2]
    frame_sequence = [1,2,3,4]

    
    classes = ["null gesture", "answer a call", "call the main menu", "exit", "confirm", "enable see-though", "shut down"]


    y_pred_list, outputs_list, y_pred_value_list = generate_online_plot(filename, save_path,max_length,window_step,num_class)

    y_pred, y_true, accuracy, frame_idx = eval_accuracy_nttd(y_pred_list, y_true, y_pred_value_list, threshold, recur_threshold,max, min)
    #
    evaluate_NTtD(classes,frame_idx,frame_sequence,y_true,y_pred)
