import base64
import copy
import datetime
import numpy as np
import socket
import sys
import time
import os
from gui_preprocess import *

import time


###################
# kgs imports
# sys.path.append(r'C:\Users\jjd50\Documents\Python\kgs_online_slim')
import GestureSpotter as kgs

from data_utils import preprocess
###################
print('\n=====================\nGestureSpotter Server\n=====================\n')

# GestureSpotter params, TODO configure in class?
detect_threshold = 0.95
recur_threshold = 5
window_step = 1
window_width = 20
num_class = 7
input_dim = 161
pos = True
quat = True
norm_to_wrist = False
relative = True

# Initialize GestureSpotter class with model
model_path = 'subject_10/model/Model'
gSpotter = kgs.GestureSpotter(model_path,input_dim)

# Output directory for logged data
output_dir = "logs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



selected_joints = ['wrist_root', 'thumb0', 'thumb1', 'thumb2', 'thumb3', 'thumbtip', 'index1', 'index2', 'index3', 'indextip', 'middle1', 'middle2', 'middle3', 'middletip', 'ring1', 'ring2', 'ring3', 'ringtip', 'pinky0', 'pinky1', 'pinky2', 'pinky3', 'pinkytip']
# TODO data sepecific params should be configured somewhere
data_buffer = np.empty((0,168), int)

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('192.168.1.185', 11001)
print('\nStarting server on %s port %s' % server_address)
sock.bind(server_address)
ip_address = socket.gethostbyname(socket.gethostname())
print('ip address: %s' % ip_address)

# Enum of message types, must match GestureSpotterClient
class MessageType:
    Invalid, Acknowledge, Goodbye, DataSample, PredictedClass = range(5)

# Listen for incoming connections
sock.listen(1)
buffer_size = 10024

while True:
    # Wait for a connection
    print('\nWaiting for a connection...')
    connection, client_address = sock.accept()
    
    try:
        print('connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(buffer_size)
            #print('\nreceived "%s"' % data)

            if data:
                splitData = data.decode().split('\t')
                if len(splitData) >= 1 and len(splitData[0]) == 1:
                    messageType = int(splitData[0])

                    # Message handling cases

                    # Acknowledge
                    if messageType == MessageType.Acknowledge:                            
                        print('\nReceived ACK message: %s' % splitData[1])
                        connection.sendall(str(int(MessageType.Acknowledge)).encode())

                    # ContextImage
                    elif messageType == MessageType.DataSample:                                                        
                        data_sample_str = splitData[1]
                        data_sample = np.fromstring(data_sample_str, dtype=float, sep=",")
                
                        # Append sample to buffer
                        # print('data buffer last 1: {}'.format(data_buffer[-1:,:]))
                        data_buffer = np.append(data_buffer, data_sample.reshape(1,168), axis=0)


                        # print('data sample: {}'.format(data_sample.reshape(1,18)))
                        # print('data buffer last 2: {}'.format(data_buffer[-1:,:]))
                        # Nominal gesture prediction and probs array
                        predicted_class = -1
                        gesture_probs = np.zeros(num_class)
                        if data_buffer.shape[0] >= window_width:
                            # Slice out data frame from buffer
                            data_quest = copy.copy(data_buffer[-window_width:, :])


                            #start_time = datetime.datetime.now()
                            data_quest = post_process(selected_joints,data_quest,pos,quat, norm_to_wrist, relative)

                            # thumb_length_list = []
                            # index_length_list = []
                            # middle_length_list = []
                            # ring_length_list = []

                            # for i in range(len(data_quest)):
                            #     thumb_length = np.sqrt(
                            #         np.power(data_quest[i][3], 2) + np.power(data_quest[i][4], 2) + np.power(
                            #             data_quest[i][5], 2))
                            #     index_length = np.sqrt(
                            #         np.power(data_quest[i][6], 2) + np.power(data_quest[i][7], 2) + np.power(
                            #             data_quest[i][8], 2))

                            #     middle_length = np.sqrt(
                            #         np.power(data_quest[i][9], 2) + np.power(data_quest[i][10], 2) + np.power(
                            #             data_quest[i][11], 2))
                            #     ring_length = np.sqrt(
                            #         np.power(data_quest[i][12], 2) + np.power(data_quest[i][13], 2) + np.power(
                            #             data_quest[i][14], 2))

                            #     thumb_length_list.append(thumb_length)
                            #     index_length_list.append(index_length)
                            #     middle_length_list.append(middle_length)
                            #     ring_length_list.append(ring_length)

                            # print('thumb: {}'.format(np.mean(thumb_length_list)))
                            # print('index: {}'.format(np.mean(index_length_list)))
                            # print('middle: {}'.format(np.mean(middle_length_list)))
                            # print('ring: {}'.format(np.mean(ring_length_list)))
                            
                            #time_preprocess = datetime.datetime.now() - start_time
                            #print('preprocess time {}'.format(time_preprocess))


                            #print(np.mean([x[2] for x in data_quest]))
                            #print(np.mean([np.mean(data_quest[i][[3, 6, 9, 12, 15]]) for i in range(len(data_quest))]))

                            data_frame = np.expand_dims(data_quest, axis=0)

                            ## 'Likihood' and recurrence thresholded prediction

                            start_time = datetime.datetime.now()

                            time.sleep(0.02)
                            spot = gSpotter.spot(data_frame, detect_threshold, recur_threshold)
                            
                            time_train = datetime.datetime.now() - start_time
                            print('prediction time {}'.format(time_train))


                            predicted_class = spot[0]
                            gesture_probs = spot[1]
                            # print(predicted_class)
                            # print(gesture_probs)
                            if predicted_class != -1:
                                print('{}) SPOTTED gesture class: {}'.format(data_buffer.shape[0], spot[0]))                                

                        # Reply with prediction message
                        # TODO this can probably be optimized to avoid replaces, just want comma separated join of values
                        gesture_probs_str = np.array2string(gesture_probs, threshold=np.inf, max_line_width=np.inf, separator=',')
                        gesture_probs_str = gesture_probs_str.replace('[','')
                        gesture_probs_str = gesture_probs_str.replace(']','')
                        prediction_msg = "%s\t%d:%s\n" % (str(int(MessageType.PredictedClass)), predicted_class, gesture_probs_str)                            
                        connection.sendall(prediction_msg.encode())
                        print(prediction_msg)
		

                    # Goodbye
                    elif messageType == MessageType.Goodbye:
                        print("Goodbye message, saving buffer")
                        output_filename = '{}/data_buffer_{}.csv'.format(output_dir, time.strftime("%Y%m%d_%H%M%S"))
                        np.savetxt(output_filename, data_buffer, fmt='%1.4f', delimiter=",")
                        # Reset data buffer
                        # TODO data sepecific params should be configured somewhere
                        data_buffer = np.empty((0,168), int)
                        connection.sendall(str(int(MessageType.Acknowledge)).encode())
                        break


            else:
                print('no more data from', client_address)
                break

    finally:
        # Clean up the connection
        connection.close()
