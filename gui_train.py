



import PySimpleGUI as sg
from eval import eval_classifier
from train import train_classifier
import os
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from datetime import datetime

import PySimpleGUI as sg
from synthesis_data import *
import os
import matplotlib.pyplot as plt

#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def prepare_log():
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Log Folder', size=(16, 1)), sg.InputText(),
               sg.FolderBrowse(button_text="Browse")],
              [sg.Submit(), sg.Cancel()]]

    window1 = sg.Window('Log Folder', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return

    log_path = values[0]

    if log_path == '':
        return
    #
    # save_path = log_path+datetime.now().strftime("/%Y%m%d-%H%M%S/")

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # log = init_logging(log_path, 'Model')

    return log_path

def parameters():
    layout = [
        [sg.Text('max length', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='20',key="max_length")
         ,
        sg.Text('batch size', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='64',key="batch_size")
         ],
        [sg.Text('input dim', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='18',key="input_dim")
         ,
        sg.Text('output dim', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='7',key="output_dim")
         ],
        [sg.Text('epochs', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='10',key="EPOCHS")
         ,
        sg.Text('learning rate', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='0.001',key="learning_rate")
         ],
        [sg.Text('drop out', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='0.4',key="drop_out")
         ],
        [sg.Submit(), sg.Cancel()]
    ]

    window1 = sg.Window('Input file', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return

    max_length = int(values['max_length'])
    batch_size = int(values['batch_size'])
    input_dim = int(values['input_dim'])
    output_dim = int(values['output_dim'])
    EPOCHS = int(values['EPOCHS'])
    learning_rate = float(values['learning_rate'])
    drop_out = float(values['drop_out'])

    return max_length, batch_size, \
        input_dim, output_dim,\
        EPOCHS, learning_rate,\
        drop_out


def pad_data(input_dim, data, max_length):
    data_padded = np.zeros([len(data), max_length, input_dim])
    for i in range(len(data)):
        if len(data[i]) <= max_length:
            data_padded[i, :len(data[i])] = data[i]
        if len(data[i]) > max_length:
            data_padded[i] = data[i][:max_length]
    return data_padded

def get_flops_(model):
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()

    run_meta = tf.compat.v1.RunMetadata()
    profiler = tf.compat.v1.profiler
    opts = profiler.ProfileOptionBuilder.float_operation()
    # We use the Keras session graph in the call to the profiler.
    flops = profiler.profile(graph=sess.graph,
                             run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model


from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

def train(max_length, batch_size,
              input_dim, output_dim,
              EPOCHS, learning_rate,
              drop_out, save_path,data_path, log, Model):

    text_write = 'max_length is {}\n batch_size is {}\n input_dim is: {}\n output_dim is: {}\n EPOCHS: {}\n learning_rate is: {}\n drop_out: {}\n save_path: {}\n data_path:{} \n Model: {}'.format(max_length, batch_size,
              input_dim, output_dim,
              EPOCHS, learning_rate,
              drop_out, save_path,data_path, Model)

    with open(save_path+'/info.txt', 'w') as f:
        f.write(text_write)

    path = data_path+'/data_aug_train.npz'
    # label = []
    # data = []

    # map_index = [11,12,16]
    # for i, fname in enumerate(sorted(glob.glob(path))):
    #     index_original = int(fname.split('.')[-2].split('_')[-1])
    #     gesture = genfromtxt(fname, delimiter=',')

    #     data.append(gesture)
    #     label.append(map_index.index(index_original))

    data_ = np.load(path)
    data =  data_['name1']
    label = data_['name2']
    print(np.shape(data))
    print(np.shape(label))
    print(data[0])
    print(label[0])


    data = pad_data(input_dim, data, max_length)

    data, label = multiplier(data, label, multi=2)

    data, label =  shuffle(data, label)

    train_data, test_data = np.split(
        data, [np.shape(data)[0] * 9 // 10]
    )
    train_label, test_label = np.split(
        label, [np.shape(label)[0] * 9 // 10]
    )

    train_data, train_label = shuffle(train_data, train_label)
    test_data, test_label = shuffle(test_data, test_label)

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    train_label = np.asarray(train_label, dtype=np.int32)
    test_label = np.asarray(test_label, dtype=np.int32)

    # select a model and set the model parameters
    model = model_selection(Model, drop_out, max_length, input_dim, output_dim)


    log.info('max_length {}'.format(max_length))
    log.info('batch_size {}'.format(batch_size))
    log.info('EPOCHS {}'.format(str(EPOCHS)))
    log.info('learning_rate {}'.format(str(learning_rate)))
    log.info('\n')

    # training


    flopnumber = get_flops_(model)

    print('flop----:{}'.format(flopnumber))

    # train_classifier(model, Model, save_path, EPOCHS,learning_rate, train_data, train_label, test_data, test_label, batch_size)


   # validating


    # path = data_path+'/data_aug_train.npz'
    #
    # data_ = np.load(path)
    # val_data =  data_['name1']
    # val_label = data_['name2']
    #
    # _,_ = eval_classifier(Model,log,save_path, val_data, val_label,batch_size)
    return


def save_model_folder():
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Model Saving Folder', size=(16, 1)), sg.InputText(),
               sg.FolderBrowse(button_text="Browse")],
              [sg.Submit(), sg.Cancel()]]

    window1 = sg.Window('Input Folder', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return

    filename = values[0]


    if filename == '':
        return



    return filename






def confusion_matrix(save_path,classes):
    cm = np.load(save_path + '/confusion_matrix.npy')

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
    plt.show()
    # plt.savefig(save_path+'/confusion_matrix.png')
    # plt.close()

if __name__ == "__main__":


    log_path ='test_1/'
    data_path = 'test_1'
    save_path = 'test_1/'
    max_length = 20
    batch_size = 64
    input_dim = 42
    output_dim = 7
    epochs = 10 
    learning_rate = 0.0001
    drop_out = 0.4
    Model = 'rnn_att_model'

    
    log_train = init_logging(log_path, 'Model')



    train(max_length, batch_size,
          input_dim, output_dim,
          epochs, learning_rate,
          drop_out, save_path, data_path, log_train, Model)


    classes = [ "null", "call", "menu", "exit", "confirm", "see-though", "shut"]
    confusion_matrix(save_path, classes)
