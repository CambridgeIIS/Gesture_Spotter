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
import datetime

from datetime import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




def init_logging_gui(log_dir):
    logging_level = logging.INFO

    log_file =datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_log.txt'

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


def read_folder():
    sg.set_options(auto_size_buttons=True)
    read_data_folder = [[sg.Text('Folder Path', size=(16, 1)), sg.InputText(),
               sg.FolderBrowse(button_text="Browse", initial_folder='/home/gesturespotter')],
              [sg.Submit(), sg.Cancel()]]

    window1 = sg.Window('Folder Finder', read_data_folder)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return

    foldername = values[0]


    if foldername == '':
        return

    return foldername
