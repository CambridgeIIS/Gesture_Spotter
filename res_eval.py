import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import editdistance


from tqdm import tqdm


def load_model(save_path):
    SAVE_PATH_model = save_path + 'Model'
    model = keras.models.load_model(SAVE_PATH_model, custom_objects={'tf': tf})
    print('loaded model from {}'.format(save_path))
    return model

def online_rec(model,max_length, window_step, input_test):
    window_width = max_length
    outputs_list = []
    y_pred_list = []
    y_pred_value_list = []

    for window in tqdm(np.arange(0, int(len(input_test) - window_width) + 1, window_step)):
        window = int(window)

        input_slice = np.expand_dims(input_test[window:window + window_width], axis=0)

        Data = tf.data.Dataset.from_tensor_slices(
            (input_slice)
        )
        Data = Data.batch(1).prefetch(buffer_size=1)

        outputs = model.predict(Data, verbose=0)
        outputs_softmax = tf.nn.softmax(outputs)
        y_pred = np.argmax(outputs_softmax, axis=1)
        y_pred_value = np.max(outputs_softmax)

        outputs_list.append(outputs_softmax)
        y_pred_list.append(int(y_pred))
        y_pred_value_list.append(float(y_pred_value))

    return y_pred_list, outputs_list, y_pred_value_list


def eval_accuracy_nttd(y_pred_list, y_true, y_pred_value_list, threshold=0.6, recur_threshold=20,max = 0.21, min = 0.08):
    pred_list = []
    y_pred_value_list_ = (np.array(y_pred_value_list) - min) / (max - min)

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

def evaluate_NTtD(classes,frame_idx,frame_sequence,y_true,y_pred):
    frame_pred = (np.array(frame_idx) + 300)[::2]
    frame_true = frame_sequence

    NTTD_status = ''
    for j, b in enumerate(y_true):
        for i, a in enumerate(y_pred):
            start = frame_true[j * 2]
            end = frame_true[j * 2 + 1]
            #         print(frame_pred[i])

            if a == b and start < frame_pred[i] < end:
                NTtD = (frame_pred[i] - start + 1) / (end - start + 1)
                NTTD_status +='{} is recognized correctly at {} seconds, with NTtD {}\n'.format(classes[b].upper(),
                                                                                      frame_pred[i] / 72, NTtD)
    return NTTD_status

def online_plot(outputs_list, input_test,input_label, frame_sequence):

    outputs_list_ = np.squeeze(outputs_list, axis=1)
    color_list_11 = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-',
                     'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--',
                     'b-.', 'g-.', 'r-.', 'c-.', 'm-.', 'y-.', 'k-.',
                     'b:', 'g:', 'r:', 'c:', 'm:', 'y:', 'k:', 'b.']

    list_ = np.zeros((12, len(input_test)))
    for i, index in enumerate(frame_sequence):
        if i % 2 == 0:
            list_[input_label[int(i / 2)]][frame_sequence[i]:frame_sequence[i + 1]] = 1

    list__ = np.transpose(list_)

    x_axis = np.arange(np.shape(outputs_list_)[0])
    x_axis_tile = np.tile(x_axis, (np.shape(outputs_list_)[1], 1))
    x_axis_tile_t = np.transpose(x_axis_tile)

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), nrows=2, sharex=True)

    for i in range(12):
        if i == 0:
            continue
        ax1.plot(np.concatenate((np.arange(300) / 72, x_axis_tile_t[:, i] / 72 + 300 / 72)),
                 np.concatenate((np.ones(300) *np.mean(outputs_list_[:, 0]), outputs_list_[:, i])), color_list_11[i])

    for i in range(12):
        if i == 0:
            continue
        ax2.plot(np.concatenate((np.arange(300) / 72, x_axis_tile_t[:-1, i] / 72 + 300 / 72)), list__[:, i],
                 color_list_11[i])

    fig.savefig('plot/online_{}.png'.format('_'.join([str(i) for i in input_label])))