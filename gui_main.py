import PySimpleGUI as sg
from gui_utils import *
from gui_preprocess import *
from gui_augmentation import *
from gui_train import *
from gui_eval import *
from gui_visual import *

font = ('Helvetica 12')
sg.set_options(font=font)

sg.theme('Default1')
layout1 = [
        [sg.Button('Load Data', size=(50, 1), enable_events=True, key='-preprocess_read-')],
     [sg.Checkbox('Relative', size=(20, 1), key='relative', default=False),
     sg.Checkbox('Hand Movement', size=(20, 1), key='norm_to_wrist', default=True)],
    [sg.Checkbox('Position', size=(20, 1), key='position', default=False),
     sg.Checkbox('Orientation', size=(20, 1), key='rotation', default=False)],
     [sg.Button('Select Joints', size=(50, 1), enable_events=True, key='-joints-')],
    [sg.Button('Output Path', size=(50, 1), enable_events=True, key='-preprocess_output-')],
    [sg.Text('_'*100) ],
     [sg.Button('Preprocess', size=(50, 1), enable_events=True, key='-preprocess-')],
[
    sg.ProgressBar(100, orientation='h', size=(50, 20), key='pre_progressbar')],
        [sg.Text("", size=(50, 10), key='-status_preprocess-', pad=(5, 5))],
]

layout2 = [
    [sg.Button('Load Data', size=(50, 1), enable_events=True, key='-aug_input-')],
    [sg.Text('Window Length', size=(50, 1), pad=(5, 5)),
     sg.InputText(default_text='20', key="max_length")
     ],
        [sg.Checkbox('Position', size=(50, 1), key='position_aug', default=False)],
     [sg.Checkbox('Rotation', size=(50, 1), key='rotation_aug', default=False)],
     [sg.Button('Augmentation Strategies', size=(50, 1), enable_events=True, key='-data_aug-')],
    [sg.Button('Output Path', size=(50, 1), enable_events=True, key='-aug_output-')],
    [sg.Text('_' * 100)],
    [sg.Button('Augment', size=(50, 1), enable_events=True, key='-Aug-')],
    [sg.ProgressBar(100, orientation='h', size=(50, 20), key='aug_progressbar')],
    [sg.Text("", size=(45, 10), key='-status_augmentation-', pad=(5, 5))],
]



model_selection = [
'SA-ResLSTM','ResNet','CNN','LSTM','Inception''smallCNN',
]

gesture_classes = ["null gesture", "answer a call", "bring up the main menu", "exit", "confirm", "enable see-though", "shut down"]



layout3 = [

[sg.Button('Log Folder', size=(30, 1), enable_events=True, key='-log-', font='Helvetica 12')],
    [
 sg.Button('Train Data Folder', size=(30, 1), enable_events=True, key='-train_data_folder-', font='Helvetica 12'),

 ],
[sg.Text('Input Classes'),sg.InputText(key='new_gesture', size=(20, 1)) ],
    [sg.Button("Add", enable_events=True, key="-add-"),
     sg.Button("Remove", enable_events=True, key="-remove-"),
     ],

    [sg.Listbox(values=gesture_classes, size=(20, 4), enable_events=True, key="-gesture_classes-")],
    [sg.Text('Model Selection')],
    [sg.Listbox(model_selection, size=(20, 4), key='LISTBOX')],
    [sg.Button('Input Parameters', size=(30, 1), enable_events=True, key='-parameters-', font='Helvetica 12')],
    [sg.Button('Model Output Path', size=(30, 1), enable_events=True, key='-model_folder-', font='Helvetica 12')],
    [sg.Text('_' * 50)],
     [sg.Button('Train', size=(10, 1), enable_events=True, key='-train-', font='Helvetica 12')],
    [sg.ProgressBar(50, orientation='h', size=(100, 20), key='train_progressbar')],
    # [
    #   sg.Button('Generate Confusion Matrix', size=(50, 1), enable_events=True, key='-confusion_matrix-', font='Helvetica 12')
    #   ],
    # [sg.Text("", size=(50, 1), key='-param_values-', pad=(5, 5), font='Helvetica 12')]

]


true_label_list = []
start_frame_sequence_list = []
end_frame_sequence_list = []


layout4 = [

[sg.Text('Test Data'),sg.InputText(size=(30, 1), key='-test_data-'), sg.FileBrowse(size=(10, 1), file_types=(("CSV files", "*.csv"),))],
[sg.Text('True Label'),sg.InputText(key='true_label', size=(5, 1)),sg.Text('Start Index'),sg.InputText(key='start_frame_sequence', size=(5, 1)),sg.Text('End Index'),sg.InputText(key='end_frame_sequence', size=(5, 1))],

    [sg.Button("Add", enable_events=True, key="-eval_add-"),
     sg.Button("Remove", enable_events=True, key="-eval_remove-"),
     ],
    [sg.Listbox(values=true_label_list, size=(20, 4), enable_events=True, key="-true_label_list-"),
     sg.Listbox(values=start_frame_sequence_list, size=(15, 4), enable_events=True, key="-start_frame_sequence_list-"),
     sg.Listbox(values=end_frame_sequence_list, size=(15, 4), enable_events=True, key="-end_frame_sequence_list-")
     ],
# [sg.Text('True Label'),sg.InputText(size=(10, 1), key='-true_label-'), sg.FileBrowse(size=(5, 1), file_types=(("CSV files", "*.csv"),))],
# [sg.Text('Frame Sequence'),sg.InputText(key='-frame_sequence-')],
    [
      sg.Button('Saved Model Folder', size=(50, 1), enable_events=True, key='-saved_model_folder-'),
    ]
    ,

[sg.Text('Number of Classes'),sg.InputText(key='-num_classes-',default_text='7')],
    [sg.Text('_' * 50)],

    [
      sg.Button('Generate Online Evaluation Plot', size=(50, 1), enable_events=True, key='-generate_online_plot-')
    ],
    [sg.Text('_' * 100)],
[sg.Text('Window Step', size=(50, 1), pad=(5, 5)),sg.InputText(default_text='1',key='-window_step-')],
        [
          sg.Text('Probability Threshold', size=(50, 1), pad=(5, 5)),
         sg.InputText(default_text='0.3',key="probability_threshold")],
          [sg.Text('Recurrence Threshold', size=(50, 1), pad=(5, 5)),
         sg.InputText(default_text='10',key="recurrence_threshold")
         ],
    # [
    #     sg.Text('Maximum Probability', size=(50, 1), pad=(5, 5)),
    #     sg.InputText(default_text='0.3', key="max")],
    # [sg.Text('Minimum Probability', size=(50, 1), pad=(5, 5)),
    #  sg.InputText(default_text='0.05', key="min")
    #  ],

    [sg.Text('_' * 50)],
     [sg.Button('Evaluate', size=(50, 1), enable_events=True, key='-evaluate-', font='Helvetica 12')],
    [sg.ProgressBar(50, orientation='h', size=(100, 20), key='progressbar')],
    [sg.Text('Recognition Accuracy:', size=(20, 1), key='-Accuracy-', pad=(5, 5), font='Helvetica 12'),
    sg.Text('', size=(20, 1), key='-accuracy_value-', pad=(5, 5), font='Helvetica 12')],
    [sg.Text('Early Detection:', size=(20, 1), key='-NTtD-', pad=(5, 5), font='Helvetica 12')],
    [sg.Text('', size=(20, 5), key='-NTTD_status-', pad=(5, 5), font='Helvetica 12'),],

]

layout5 = [
        [sg.Text('Load Data'),sg.InputText(size=(40, 1), key='-visual_data-'), sg.FileBrowse(size=(10, 1), file_types=(("CSV files", "*.csv"),))],
    [sg.Text('_' * 100)],
     [
      sg.Button( 'Visualize', size=(50, 1), enable_events=True, key='-visualization-')
    ],

]

preprocess_tooltip = 'Preprocess the data: ' \
                     '\nload data: enter the data folder where the data need to be processed ' \
                     '\nRelative: make absolute position trajectories to relative trajectories' \
                     '\nNorm to palm: make other joints data normalize to palm' \
                     '\nPosition: take position values' \
                     '\nRotation: take quaternion values' \
                     '\nSelect Joints: select the joints to be included in the processed data' \
                     '\nPreprocess: process the raw data according to the conditions entered above'


augmentation_tooltip = 'Augmenting the data:' \
                       '\nload data: enter the data folder where the data need to be augmented' \
                       '\nAugmentation Strategy: select the augmentation strategies' \
                       '\n--- GAN AUG: check if GAN augmentation is needed' \
                       '\n--- Noise Parameter: the parameter in the noise augmentation strategy, ' \
                       '\nif entered 0, then noise augmentation strategy is not used' \
                       '\n--- Shift Parameter: the parameter in the shift augmentation strategy, ' \
                       '\nif entered 0, then shift augmentation strategy is not used' \
                       '\n--- Scale Parameter: the parameter in the scale augmentation strategy,' \
                       '\nif entered 0, then scale augmentation stratgy is not used' \
                       '\noutput path: enetr the data flder where the augmented data will be stored' \
                       '\nprocess: start the augmentation process'

train_tooltip = 'train the data:' \
                '\nLog Folder:' \
                '\nTrain Data Folder:' \
                '\nInput Class: enter the class, seperated by ;(for example, a;b;c)' \
                '\nModel Selection:'
layout = [[sg.TabGroup([[sg.Tab('Preprocessing', layout1,
                                tooltip=preprocess_tooltip, element_justification= 'center'),
                    sg.Tab('Augmenting', layout2, tooltip=augmentation_tooltip,element_justification= 'center'),
                    sg.Tab('Training', layout3,
                           tooltip=train_tooltip, element_justification= 'center')
                           ,
                    sg.Tab('Evaluating', layout4,
                           tooltip='Evaluation', element_justification= 'center')
                                                   ,
                    sg.Tab('Visualizing', layout5,
                           tooltip='Visualization', element_justification= 'center')

                           ]], tab_location='centertop', border_width=0), sg.Button('Close')]]
        

window = sg.Window('GestureSpotter', layout, size=(600, 800))



main_project_folder = input("Enter your main folder: ")
# main_project_folder = 'subject_4/'

if not os.path.exists(main_project_folder):
    os.makedirs(main_project_folder)
    
log = init_logging_gui(main_project_folder)

# log.info('max_length {}'.format(max_length))
# log.info('batch_size {}'.format(batch_size))
# log.info('EPOCHS {}'.format(str(EPOCHS)))
# log.info('learning_rate {}'.format(str(learning_rate)))
# log.info('\n')

#######preprocess#######
preprocess_status = ''
proprocess_folder = ''
selected_joints = []
relative = False
norm_to_wrist = False
rotation = False
position = False
pre_progressbar = window['pre_progressbar']
status_preprocess = window['-status_preprocess-']


#######augmentation#######
fn = ''
aug_progress_bar = window['aug_progressbar']
shift_param = 0
noise_param = 0
scale_param = 0
gan_aug = False
status_augmentation = window['-status_augmentation-']

input_path = ''
output_path = ''

compoent_num = 6

rotation_aug = False
position_aug = False

#######train#######
save_path = ''
data_path = ''
log_train = ''
Model= 'resNet'
max_length = 20
batch_size = 64
input_dim = 42
output_dim = 17
EPOCHS = 100
learning_rate = 0.0001
drop_out = 0.4
# param_values = window['-param_values-']
train_progress_bar = window['train_progressbar']


#######eval#######
classes = ''
saved_model_folder = ''
y_pred_list = []
y_pred_value_list = []
classes = ["null gesture", "answer a call", "call the main menu", "exit", "confirm", "enable see-though", "shut down"]
NTTD_status = window['-NTTD_status-']



accuracy_value = window['-accuracy_value-']
while True:
    event, values = window.read()


    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    
    #######preprocess#######
    if event == '-preprocess_read-':
        log.info(event+'\n')

        proprocess_folder = read_folder()
        current_status = 'your selected input folder path is : {}'.format(proprocess_folder)
        preprocess_status = preprocess_status + current_status
        # status_preprocess.update(preprocess_status)

        
        log.info(current_status+'\n')


    if event == '-joints-':
        log.info(event+'\n')
        
        selected_joints = joints()
        current_status = 'your selected joints are : {}'.format(selected_joints)
        preprocess_status = preprocess_status + current_status
        compoent_num = len(selected_joints)
        # status_preprocess.update(preprocess_status)

        
        log.info(current_status+'\n')

    if event == '-preprocess_output-':
        log.info(event+'\n')

        proprocess_output_folder = read_folder()
        current_status  = ' your selected output folder path is : {}'.format(proprocess_output_folder)
        preprocess_status = preprocess_status + current_status
                                                
        # status_preprocess.update(preprocess_status)
        log.info(current_status+'\n')

    if event == '-preprocess-':

        if values['relative'] == True:
            log.info('relative == True' + '\n')
            relative = True

        if values['norm_to_wrist'] == False:
            log.info('norm_to_wrist == True' + '\n')
            norm_to_palm = True

        if values['rotation'] == True:
            log.info('rotation == True' + '\n')
            rotation = True

        if values['position'] == True:
            log.info('position == True' + '\n')
            position = True

        if values['relative'] == False:
            log.info('relative == False' + '\n')
            relative = False

        if values['norm_to_wrist'] == True:
            log.info('norm_to_wrist == False' + '\n')
            norm_to_palm = False

        if values['rotation'] == False:
            log.info('rotation == False' + '\n')
            rotation = False

        if values['position'] == False:
            log.info('position == False' + '\n')
            position = False


        gui_preprocess(proprocess_folder,proprocess_output_folder,relative,norm_to_wrist,rotation,position,selected_joints)
        preprocess_status = preprocess_status + '------------preprocessing'

        for i in range(100):
           event, values = window.read(timeout=10)
           pre_progressbar.UpdateBar(i + 1)

        preprocess_status = preprocess_status + 'finished'
       # status_preprocess.update(preprocess_status)
    ##########################



    #######augmentation#######
    if event == '-aug_input-':
        log.info(event+'\n')

        input_path = read_folder()
        current_status = 'the augmented data input path is :{}'.format(input_path)
        log.info(current_status+'\n')

        


    if event == '-data_aug-':
        log.info(event+'\n')

        shift_param, noise_param, scale_param, gan_aug = gui_augmentation()
        current_status = 'shift parameter:{}, noise parameter:{}, scale parameter:{}, GAN Augmentation '.format(shift_param, noise_param, scale_param, gan_aug)
        log.info(current_status+'\n')
        # status_augmentation.update('your parameters chosen are: shift: {}, noise: {}, scale: {}'.format(shift_param, noise_param, scale_param))
    if event == '-aug_output-':
        log.info(event+'\n')

        output_path = read_folder()
        current_status = 'the augmented data output path is :{}'.format(output_path)
        log.info(current_status+'\n')

    if event == '-Aug-':

        if values['rotation_aug'] == True:
            log.info('rotation_aug == True' + '\n')
            rotation = True

        if values['position_aug'] == True:
            log.info('position_aug == True' + '\n')
            position = True

        if values['rotation_aug'] == False:
            log.info('rotation_aug == False' + '\n')
            rotation = False
        if values['position_aug'] == False:
            log.info('position_aug == False' + '\n')
            position = False

        log.info(event+'\n')

        # status_augmentation.update("data augmenting...")
        max_length = int(values['max_length'])
        ges_aug(max_length,input_path,output_path,compoent_num,shift_param, noise_param, scale_param, gan_aug,position_aug,rotation_aug)
        for i in range(100):
            event, values = window.read(timeout=10)
            aug_progress_bar.UpdateBar(i + 1)


    ##########################


    #######train#######
    if event == '-log-':
        log.info(event+'\n')

        log_path = prepare_log()
        log_path = log_path +'/'
        log_train = init_logging(log_path, 'Model')

    if event == '-train_data_folder-':
        log.info(event+'\n')

        data_path = read_folder()
        current_status = 'the train data path is :{}'.format(data_path)

        log.info(current_status+'\n')

    if event == "-add-":
        print(values['new_gesture'])
        new_gesture = values['new_gesture']
        gesture_classes.append(new_gesture)
        window["-gesture_classes-"].update(gesture_classes)


    if event == "-remove-":
        new_gesture = values['-gesture_classes-'][0]

        INDEX = int(gesture_classes.index(new_gesture))
        gesture_classes.pop(INDEX)
        window["-gesture_classes-"].update(gesture_classes)

    if event == '-model_folder-':
        log.info(event+'\n')

        save_path = save_model_folder()
        current_status = 'the model path is :{}'.format(save_path)
        save_path = save_path +'/'

        log.info(current_status+'\n')


    if event == '-parameters-':
        log.info(event+'\n')

        max_length, batch_size, \
        input_dim, output_dim,\
        epochs, learning_rate,\
        drop_out = parameters()



        # param_values.update("Input Dim: {}".format(input_dim))

    if event == '-train-':
        log.info(event+'\n')

        Model = values['LISTBOX'][0]
        
        current_status = 'log_path = {}\ndata_path = {}\nsave_path = {}\nmax_length ={}\nbatch_size={}\ninput_dim={}\noutput_dim={}\nepochs={}\nlearning_rate={}\ndrop_out={}\nModel = {}'.format(log_path,data_path,save_path, max_length,batch_size,input_dim,output_dim,epochs,learning_rate,drop_out,Model)

        log.info(current_status+'\n')

        train(max_length, batch_size,
              input_dim, output_dim,
              epochs, learning_rate,
              drop_out, save_path,data_path, log_train, Model)


    # if event == '-confusion_matrix-':
    #     classes = values['-classes-'].split(';')
        confusion_matrix(save_path,gesture_classes)

    ##########################



    #######eval#######
    if event == '-saved_model_folder-':
        log.info(event+'\n')

        saved_model_folder = read_folder()

        current_status = 'the model path is :{}'.format(saved_model_folder)

        log.info(current_status+'\n')

    if event == "-eval_add-":
        true_label = values['true_label']
        true_label_list.append(true_label)
        window["-true_label_list-"].update(true_label_list)

        start_frame_sequence = values['start_frame_sequence']
        start_frame_sequence_list.append(start_frame_sequence)
        window["-start_frame_sequence_list-"].update(start_frame_sequence_list)

        end_frame_sequence = values['end_frame_sequence']
        end_frame_sequence_list.append(end_frame_sequence)
        window["-end_frame_sequence_list-"].update(end_frame_sequence_list)


    if event == "-eval_remove-":
        true_label_rm = values['-true_label_list-'][0]
        INDEX = int(true_label_list.index(true_label_rm))
        true_label_list.pop(INDEX)
        window["-true_label_list-"].update(true_label_list)


        start_frame_sequence_rm= values['-start_frame_sequence_list-'][0]
        INDEX = int(start_frame_sequence_list.index(start_frame_sequence_rm))
        start_frame_sequence_list.pop(INDEX)
        window["-start_frame_sequence_list-"].update(start_frame_sequence_list)

        end_frame_sequence_rm= values['-end_frame_sequence_list-'][0]
        INDEX = int(end_frame_sequence_list.index(end_frame_sequence_rm))
        end_frame_sequence_list.pop(INDEX)
        window["-end_frame_sequence_list-"].update(end_frame_sequence_list)


    if event == '-generate_online_plot-':
        log.info(event+'\n')


        filename = values['-test_data-']
        num_class = int(values['-num_classes-'])
        window_step = int(values['-window_step-'])
        y_pred_list, outputs_list, y_pred_value_list = generate_online_plot(gesture_classes,filename,saved_model_folder,max_length,window_step,num_class)

        current_status = 'the model path is :{}'.format(saved_model_folder)

        log.info(current_status+'\n')

    if event == '-evaluate-':

        log.info(event+'\n')


        y_true = [int(x) for x in true_label_list]
        filename = values['-test_data-']

        num_class = int(values['-num_classes-'])
        window_step = int(values['-window_step-'])

        probabilityt_threshold = float(values['probability_threshold'])
        recurrence_threshold = float(values['recurrence_threshold'])
        # max = float(values['max'])
        # min = float(values['min'])
        frame_sequence_list = []
        for i in range(len(true_label_list)):
            frame_sequence_list.append(int(start_frame_sequence_list[i]))
            frame_sequence_list.append(int(end_frame_sequence_list[i]))


        # y_pred_list, outputs_list, y_pred_value_list = generate_online_plot(filename, saved_model_folder, max_length,
        #                                                                     window_step, num_class)

        y_pred, y_true, accuracy, frame_idx = eval_accuracy_nttd_(y_pred_list, y_true, y_pred_value_list, probabilityt_threshold,
                                                                 recurrence_threshold)
        #
        NTTD = evaluate_NTtD(classes, frame_idx, frame_sequence_list, y_true, y_pred)


        accuracy_value.update('{}'.format(accuracy))
        NTTD_status.update('\n{}'.format(NTTD))

        current_status = 'the accuracy is :{}\n {} '.format(accuracy,NTTD)
        log.info(current_status+'\n')

    if event == '-visualization-':
        input_data_path = values['-visual_data-']
        visualize(input_data_path)

window.close()
