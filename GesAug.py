import PySimpleGUI as sg
from synthesis_data import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_folder():
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


def ges_aug(fn,noise_aug, gan_aug):

    # update +++++++++
    num_synthetic_data = 20
    data_path = fn+'/*.csv'
    path_to_saved_model = '/home/shawn/kgs_online_slim/generator_g_1'
    save_path = '/home/shawn/kgs_online_slim/synthetic_data/'
    # update +++++++++
    synthesis_data(num_synthetic_data, data_path, path_to_saved_model, save_path)



    return

def joints():
    layout = [
        [sg.Checkbox('GAN Aug', size=(10, 1), key='gan', default=True)
         ,
        sg.Checkbox('GAN Aug', size=(10, 1), key='gan', default=True)
         ],
        [sg.Checkbox('GAN Aug', size=(10, 1), key='gan', default=True)]
         ,
        [sg.Checkbox('Pinky', size=(10, 1), key='gan', default=True)
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



#

layout = [
    [sg.Button('Load data', size=(10, 1), enable_events=True, key='-READ-', font='Helvetica 16')],
    [sg.Text("", size=(50, 1), key='-filename-', pad=(5, 5), font='Helvetica 12')],
    [sg.Checkbox('Noise Aug', size=(10, 1), key='noise', default=True),
     sg.Checkbox('GAN Aug', size=(10, 1), key='gan', default=True)],
     [sg.Checkbox('Relative', size=(10, 1), key='noise', default=True),
     sg.Checkbox('Norm to palm', size=(10, 1), key='gan', default=True)],
     [sg.Button('Select Joints', size=(10, 1), enable_events=True, key='-joints-', font='Helvetica 16')],
    [sg.Button('Aug Them!', size=(10, 1), enable_events=True, key='-Aug-', font='Helvetica 16')],
    [
        sg.Text("", size=(50, 1), key='-status-', pad=(5, 5), font='Helvetica 12'),
        sg.ProgressBar(100, orientation='h', size=(200, 20), key='progressbar')]
]

# Create the window
window = sg.Window('GesAug', layout, size=(600, 300))

filename = window['-filename-']
status = window['-status-']
progress_bar = window['progressbar']

noise_aug = False
gan_aug = False

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == '-READ-':
        fn = read_folder()
        filename.update("The Folder Name is: {}".format(fn))
        read_successful = True
        if read_successful:
            if values['noise'] == True:
                noise_aug = True
            if values['gan'] == True:
                gan_aug = True
    if event == '-joints-':
        joints()

    if event == '-Aug-':
        status.update("data augmenting...")
        for i in range(100):
            event, values = window.read(timeout=10)
            progress_bar.UpdateBar(i + 1)
        ges_aug(fn,noise_aug, gan_aug)
        status.update("completed augmenting...")

window.close()