
import PySimpleGUI as sg
from synthesis_data import *
import numpy as np
from numpy import genfromtxt
import random

def gui_augmentation():
    data_augmentation = [
                [sg.Checkbox('GAN Augmentation', size=(10, 1), key='gan_aug', default=False)]
         ,
        [sg.Text('noise parameter', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='0.1',key="noise")]
         ,
        [sg.Text('shift parameter', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='0.1',key="shift")
         ],
        [sg.Text('scale parameter', size=(50, 1), pad=(5, 5), font='Helvetica 12'),
         sg.InputText(default_text='0.2',key="scale")
         ],

        [sg.Submit(), sg.Cancel()]
    ]

    window1 = sg.Window('Input Parameters', data_augmentation)

    gan_aug = False
    try:
        event, values = window1.read()

        if values['gan_aug'] == True:
            gan_aug = True


        window1.close()
    except:
        window1.close()
        return

    noise = float(values['noise'])
    shift = float(values['shift'])
    scale = float(values['scale'])

    return noise, shift, scale, gan_aug


def data_aug(skeleton, compoent_num, noise, shift, scale):
    joints = 3
    do_noise = False
    do_scale = False
    do_shift = False


    if noise !=0:
        do_noise = True
    if scale !=0:
        do_scale = True
    if shift !=0:
        do_shift = True

    def scale(skeleton):
        ratio = 0.2
        low = 1 - ratio
        high = 1 + ratio
        factor = np.random.uniform(low, high)
        video_len = skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(compoent_num):
                skeleton[t][j_id] *= factor
        skeleton = np.array(skeleton)
        return skeleton

    def shift(skeleton):
        low = -0.1
        high = -low
        offset = np.random.uniform(low, high, joints)
        video_len = skeleton.shape[0]
        for t in range(video_len):
            for j_id in range(compoent_num):
                skeleton[t][j_id] += offset
        skeleton = np.array(skeleton)
        return skeleton

    def noise(skeleton):

        high = 0.1
        low = -high
        #select 4 joints
        all_joint = list(range(compoent_num))
        random.shuffle(all_joint)
        selected_joint = all_joint[0:4]

        for j_id in selected_joint:
            noise_offset = np.random.uniform(low, high, joints)
            for t in range(len(skeleton)):
                skeleton[t][j_id] += noise_offset
        skeleton = np.array(skeleton)
        return skeleton


    skeleton = np.array(skeleton).reshape(len(skeleton),compoent_num,joints)
    if do_noise:
        skeleton = noise(skeleton)
    if do_scale:
        skeleton = scale(skeleton)
    if do_shift:
        skeleton = shift(skeleton)
    # skeleton -= skeleton[0][1]
    skeleton = np.array(skeleton).reshape(len(skeleton),compoent_num*joints)

    return skeleton


def ges_aug(max_length,input_path,output_path, compoent_num,shift_param, noise_param, scale_param, gan_aug,position,rotation):
    num_synthetic_data = 20
    # update +++++++++
    data_path = input_path+'/*.csv'
    path_to_saved_model = '/home/shawn/kgs_online_slim/generator_g_1'
    # update +++++++++


    skeleton_list_train = []  
    label_train = []    

    index=0

    skeleton_lis_test = []
    label_test = []

    for _, fname in enumerate(sorted(glob.glob(data_path))):
        class_idx = int(fname.split('.')[-2][-1])
        data_ = genfromtxt(fname, delimiter=',')
        step = 5
        for i in np.arange(0, int(len(data_) - max_length) + 1, step):
                data = data_[i:i+max_length]
                index+=1
                skeleton = data
                if position:
                    if not rotation:
                        skeleton = data_aug(data, compoent_num, noise_param, shift_param, scale_param)
                    else:
                        pos = data[:,:compoent_num*3]
                        rot = data[:,compoent_num*3:]
                        pos_aug = data_aug(pos, compoent_num, noise_param, shift_param, scale_param)
                        skeleton = np.concatenate([pos_aug,rot],axis =1)

                if(index%10==0):
                    skeleton_lis_test.append(data)
                    label_test.append(class_idx)
                else:
                    skeleton_list_train.append(skeleton)
                    label_train.append(class_idx)



    # if gan_aug:
    #     skeleton_list, label = synthesis_data(num_synthetic_data, data_path, path_to_saved_model, output_path,max_length,np.shape(skeleton_list)[2])

    output_filename_test = '{}/data_aug_test.npz'.format(output_path)
    print(np.shape(skeleton_lis_test))
    np.savez(output_filename_test, name1=skeleton_lis_test, name2=label_test)

    output_filename_test = '{}/data_aug_train.npz'.format(output_path)
    print(np.shape(skeleton_list_train))
    np.savez(output_filename_test, name1=skeleton_list_train, name2=label_train)