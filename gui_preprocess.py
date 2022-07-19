from xml.etree.ElementPath import prepare_parent
from data_utils import preprocess
from numpy import genfromtxt
import glob
import numpy as np
import PySimpleGUI as sg

def joints():
    joints = [
        [sg.Checkbox('Wrist Root', size=(10, 1), key='wrist_root', default=True)]
        ,
        [sg.Checkbox('Thumb0', size=(10, 1), key='thumb0', default=True),
         sg.Checkbox('Thumb1', size=(10, 1), key='thumb1', default=True),
         sg.Checkbox('Thumb2', size=(10, 1), key='thumb2', default=True),
         sg.Checkbox('Thumb3', size=(10, 1), key='thumb3', default=True),
         sg.Checkbox('ThumbTip', size=(10, 1), key='thumbtip', default=True)
         ],
        [sg.Checkbox('Index1', size=(10, 1), key='index1', default=True),
         sg.Checkbox('Index2', size=(10, 1), key='index2', default=True),
         sg.Checkbox('Index3', size=(10, 1), key='index3', default=True),
         sg.Checkbox('IndexTip', size=(10, 1), key='indextip', default=True),
         ],
        [sg.Checkbox('Middle1', size=(10, 1), key='middle1', default=True),
         sg.Checkbox('Middle2', size=(10, 1), key='middle2', default=True),
         sg.Checkbox('Middle3', size=(10, 1), key='middle3', default=True),
         sg.Checkbox('MiddleTip', size=(10, 1), key='middletip', default=True),
         ],
        [sg.Checkbox('Ring1', size=(10, 1), key='ring1', default=True),
         sg.Checkbox('Ring2', size=(10, 1), key='ring2', default=True),
         sg.Checkbox('Ring3', size=(10, 1), key='ring3', default=True),
         sg.Checkbox('RingTip', size=(10, 1), key='ringtip', default=True),
         ],
        [sg.Checkbox('Pinky0', size=(10, 1), key='pinky0', default=True),
         sg.Checkbox('Pinky1', size=(10, 1), key='pinky1', default=True),
         sg.Checkbox('Pinky2', size=(10, 1), key='pinky2', default=True),
         sg.Checkbox('Pinky3', size=(10, 1), key='pinky3', default=True),
         sg.Checkbox('PinkyTip', size=(10, 1), key='pinkytip', default=True),
         ],
        [sg.Text("", size=(45, 10), key='-num_joints-', pad=(5, 5))],
        [sg.Submit(), sg.Cancel()]
    ]

    window1 = sg.Window('Input file', joints)

    wrist_root = False

    thumb0 = True
    thumb1 = True
    thumb2 = True
    thumb3 = True
    thumb4 = True
    thumbtip = True

    index1 = True
    index2 = True
    index3 = True
    indextip = True

    middle1 = True
    middle2 = True
    middle3 = True
    middletip = True

    ring1 = True
    ring2 = True
    ring3 = True
    ringtip = True

    pinky0 = True
    pinky1 = True
    pinky2 = True
    pinky3 = True
    pinky4 = True
    pinkytip = True

    selected_joints =  ['wrist_root', 'thumb0', 'thumb1', 'thumb2', 'thumb3','thumbtip', 'index1', 'index2', 'index3', 'indextip', 'middle1',
                        'middle2', 'middle3', 'middletip', 'ring1', 'ring2', 'ring3', 'ringtip', 'pinky0', 'pinky1','pinky2',
                        'pinky3','pinkytip']

    num_joint = 0
    try:
        event, values = window1.read()

        if values['wrist_root'] == True:
            selected_joints.append('wrist_root')
            num_joint+=1
        else:
            selected_joints.remove('wrist_root')

            # window1["-num_joints-"].update(string(num_joint))

        if values['thumb0'] == True:
            selected_joints.append('thumb0')
            num_joint += 1
        else:
            selected_joints.remove('thumb0')

        if values['thumb1'] == True:
            selected_joints.append('thumb1')
            num_joint += 1
        else:
            selected_joints.remove('thumb1')

        if values['thumb2'] == True:
            selected_joints.append('thumb2')
            num_joint += 1
        else:
            selected_joints.remove('thumb2')

        if values['thumb3'] == True:
            selected_joints.append('thumb3')
            num_joint += 1
        else:
            selected_joints.remove('thumb3')

        if values['thumbtip'] == True:
            selected_joints.append('thumbtip')
            num_joint += 1
        else:
            selected_joints.remove('thumbtip')


        if values['index1'] == True:
            selected_joints.append('index1')
            num_joint += 1
        else:
            selected_joints.remove('index1')


        if values['index2'] == True:
            selected_joints.append('index2')
            num_joint += 1
        else:
            selected_joints.remove('index2')

        if values['index3'] == True:
            selected_joints.append('index3')
            num_joint += 1
        else:
            selected_joints.remove('index3')

        if values['indextip'] == True:
            selected_joints.append('indextip')
            num_joint += 1
        else:
            selected_joints.remove('indextip')

        if values['middle1'] == True:
            selected_joints.append('middle1')
            num_joint += 1
        else:
            selected_joints.remove('middle1')


        if values['middle2'] == True:
            selected_joints.append('middle2')
            num_joint += 1
        else:
            selected_joints.remove('middle2')


        if values['middle3'] == True:
            selected_joints.append('middle3')
            num_joint += 1
        else:
            selected_joints.remove('middle3')


        if values['middletip'] == True:
            selected_joints.append('middletip')
            num_joint += 1
        else:
            selected_joints.remove('middletip')

        if values['ring1'] == True:
            selected_joints.append('ring1')
            num_joint += 1
        else:
            selected_joints.remove('ring1')

        if values['ring2'] == True:
            selected_joints.append('ring2')
            num_joint += 1
        else:
            selected_joints.remove('ring2')

        if values['ring3'] == True:
            selected_joints.append('ring3')
            num_joint += 1
        else:
            selected_joints.remove('ring3')



        if values['ringtip'] == True:
            selected_joints.append('ringtip')
            num_joint += 1
        else:
            selected_joints.remove('ringtip')


        if values['pinky0'] == True:
            selected_joints.append('pinky0')
            num_joint += 1
        else:
            selected_joints.remove('pinky0')


        if values['pinky1'] == True:
            selected_joints.append('pinky1')
            num_joint += 1
        else:
            selected_joints.remove('pinky1')


        if values['pinky2'] == True:
            selected_joints.append('pinky2')
            num_joint += 1
        else:
            selected_joints.remove('pinky2')


        if values['pinky3'] == True:
            selected_joints.append('pinky3')
            num_joint += 1
        else:
            selected_joints.remove('pinky3')


        if values['pinkytip'] == True:
            selected_joints.append('pinkytip')
            num_joint += 1
        else:
            selected_joints.remove('pinkytip')

        # window1["-num_joints-"].update(string(num_joint))

        window1.close()

    except:
        window1.close()

        return
    selected_joints = list(dict.fromkeys(selected_joints))
    return selected_joints

def select_joint(selected_joint, gesture_data, pos=True, quat=False):

                                    
    split_joint_name = ['wrist_root', 'thumb0', 'thumb1', 'thumb2', 'thumb3','thumbtip', 'index1', 'index2', 'index3', 'indextip', 'middle1',
                        'middle2', 'middle3', 'middletip', 'ring1', 'ring2', 'ring3', 'ringtip', 'pinky0', 'pinky1','pinky2',
                        'pinky3','pinkytip']

    selected_joint_idx = []
    for x in selected_joint:
        selected_joint_idx.append(split_joint_name.index(x)+1)

    idx_list_pos_x = np.multiply(selected_joint_idx, 7)
    idx_list_pos_y = np.multiply(selected_joint_idx, 7) + 1
    idx_list_pos_z = np.multiply(selected_joint_idx, 7) + 2
    idx_list_quat_x = np.multiply(selected_joint_idx, 7) + 3
    idx_list_quat_y = np.multiply(selected_joint_idx, 7) + 4
    idx_list_quat_z = np.multiply(selected_joint_idx, 7) + 5
    idx_list_quat_w = np.multiply(selected_joint_idx, 7) + 6

    idx_list_pos = np.concatenate([idx_list_pos_x, idx_list_pos_y, idx_list_pos_z])
    idx_list_quat = np.concatenate([idx_list_quat_x, idx_list_quat_y, idx_list_quat_z, idx_list_quat_w])

    position = gesture_data[:, sorted(idx_list_pos)]
    rotation = gesture_data[:, sorted(idx_list_quat)]
    
    if pos:
        if not quat:
            data = position
        else:
            data = [position, rotation]
    if quat:
        if not pos:
            data = rotation
    # if pos and quat:
        # data = gesture_data[:, sorted(np.concatenate([idx_list_pos, idx_list_quat]))]
        # data = np.concatenate([position, rotation],axis = 1)
        # data = [position, rotation]
    return data


def post_process(selected_joints,new_gesture,pos,quat, norm_to_wrist, relative):
    new_gesture = select_joint(selected_joints, new_gesture, pos, quat)


    length_joints = len(selected_joints)
    xlist = [i*3 for i in range(length_joints)]
    # xlist = [[0, 3, 6, 9, 12, 15]]
    ylist = np. add(xlist ,1)
    zlist = np. add(xlist ,2)

    if pos:
        if not quat:
            new_gesture = preprocess(new_gesture, xlist,norm_to_wrist, relative)
            print('select joints {}'.format(np.shape(new_gesture)))
        if quat:
            position = new_gesture[0]
            rotation = new_gesture[1]
            new_position = preprocess(position, xlist,norm_to_wrist, relative)
            new_gesture = np.concatenate([new_position, rotation],axis = 1)

    if quat:
        if not pos:
            new_gesture = new_gesture

    
    return new_gesture

def gui_preprocess(proprocess_folder,output_folder, relative,norm_to_wrist,pos,quat,selected_joints):
    data_path = proprocess_folder+'/*.csv'

    text_write = 'processed folder is {}\n output folder is {}\n relative is: {}\n norm to wrist is: {}\n post is: {}\n quat is: {}\n selected joints are: {}\n'.format(proprocess_folder,output_folder, relative,norm_to_wrist,pos,quat,selected_joints)

    with open(output_folder+'/info.txt', 'w') as f:
        f.write(text_write)

    for i, fname in enumerate(sorted(glob.glob(data_path))):
        new_gesture = genfromtxt(fname, delimiter=',')
        new_gesture = post_process(selected_joints,new_gesture,pos,quat,norm_to_wrist, relative)

        file_name = fname.split('_')[-1]
        print(file_name)
        output_filename = output_folder+'/'+file_name
        print(output_filename)
        np.savetxt(output_filename, new_gesture, fmt='%1.4f', delimiter=",")

