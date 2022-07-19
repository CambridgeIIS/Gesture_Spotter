import PySimpleGUI as sg
from eval import eval_classifier
from train import train_classifier
import os
import tensorflow as tf
import numpy as np
from data_utils import *
from functools import partial
from datetime import datetime



import os
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

    save_path = log_path+datetime.now().strftime("%Y%m%d-%H%M%S/")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log = init_logging(save_path, 'Model')

    return log

def parameters():
    layout = [
        [sg.Text('max length', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='300',key="max_length")
         ,
        sg.Text('batch size', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='64',key="batch_size")
         ],
        [sg.Text('input dim', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='18',key="input_dim")
         ,
        sg.Text('output dim', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='12',key="output_dim")
         ],
        [sg.Text('epochs', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='100',key="EPOCHS")
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


def train(max_length, batch_size,
              input_dim, output_dim,
              EPOCHS, learning_rate,
              drop_out, save_path,data_path, log):
    data = np.load(data_path+'/train_data.npy', allow_pickle=True)
    label = np.load(data_path+'/train_label.npy', allow_pickle=True)



    # update ++++++++
    new_path = '/home/shawn/kgs_online_slim/data/logs'
    # update ++++++++

    path = new_path+'/*.csv'
    new_gesture_list , new_label_list = get_new_gesture_data(path,max_length,input_dim, data_synthesis = True)


    # data = denoise_all(data,input_dim)
    data = interpolate_all(data,max_length,input_dim)


    train_data, train_label, test_data, test_label = train_test_split(data, label, new_gesture_list, new_label_list, 20)

    # select a model and set the model parameters
    model = model_selection(Model, drop_out, max_length, input_dim, output_dim)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # log info

    log.info('max_length {}'.format(max_length))
    log.info('batch_size {}'.format(batch_size))
    log.info('EPOCHS {}'.format(str(EPOCHS)))
    log.info('learning_rate {}'.format(str(learning_rate)))
    log.info('\n')

    # training
    train_classifier(model, Model, save_path, EPOCHS,
                     learning_rate, train_data, train_label, test_data, test_label, batch_size)

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



def read_data_folder():
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Dataset Folder', size=(16, 1)), sg.InputText(),
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



layout = [
[sg.Button('Log Folder', size=(10, 1), enable_events=True, key='-log-', font='Helvetica 16'),
 sg.Button('Data Folder', size=(10, 1), enable_events=True, key='-data_folder-', font='Helvetica 16'),
 ],
    [sg.Button('Input Parameters', size=(10, 1), enable_events=True, key='-parameters-', font='Helvetica 16')],
    [sg.Button('model output path', size=(50, 1), enable_events=True, key='-model_folder-', font='Helvetica 12'),
     sg.Button('Train it', size=(10, 1), enable_events=True, key='-train-', font='Helvetica 16')],
    [sg.ProgressBar(50, orientation='h', size=(100, 20), key='progressbar')],
    [sg.Text("", size=(50, 1), key='-param_values-', pad=(5, 5), font='Helvetica 12')]
]

window = sg.Window('Pima', layout, size=(600, 300))
progress_bar = window['progressbar']
param_values = window['-param_values-']

# select the model
Model= 'resNet'
# set model and data parameters
max_length = 300
batch_size = 64
input_dim = 18
output_dim = 12
EPOCHS = 100
learning_rate = 0.0001
drop_out = 0.4
# update ++++++++
# name the new gesture
new_gesture_name = 'horns'
# update ++++++++

while True:
    event, values = window.read()

    if event == '-log-':
        log = prepare_log()
    if event == '-data_folder-':
        data_path = read_data_folder()
    if event == '-model_folder-':
        save_path = save_model_folder()
    if event == '-parameters-':
        max_length, batch_size, \
        input_dim, output_dim,\
        EPOCHS, learning_rate,\
        drop_out = parameters()

        param_values.update("Input Dim: {}".format(input_dim))

    if event == '-train-':
        train(max_length, batch_size,
              input_dim, output_dim,
              EPOCHS, learning_rate,
              drop_out, save_path,data_path, log)



