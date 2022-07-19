import numpy as np
import os
import socket
import sys
import time
from pynput import keyboard

###################
print('\n==========================\nGestureSpotter Test Client\n==========================\n')

###
# DEV TEST DATA
dataset = np.load('data/online.npz',allow_pickle = True)
data = dataset['arr_0']
idx = 20
input_test = data[idx]

###################
# Get server details
ip_address = input("Enter an ip address: ")
port = input("Enter port: ")

# Create a TCP/IP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = (ip_address, int(port))
print('Connect to server at %s port %s' % server_address)
s.connect(server_address)
s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 100)

# Enum of message types, must match GestureSpotterSever
class MessageType:
    Invalid, Acknowledge, Goodbye, DataSample, PredictedClass = range(5)

# Buffer size
buffer_size = 1024

try:
    while True:
        # Wait for a message    
        msg = input("Enter message: ")
        
        if (msg == "exit"):
            msg = "2"
            s.send(msg.encode('utf-8'))
            break         
        elif (msg == "connection_test"):
            print('\nSend connection test msg.')
            len_msg = "%s\t%d\t" % (str(int(MessageType.Acknowledge)), 123)            
            s.send(len_msg.encode('utf-8'))
            time.sleep(.100)
            print('Received: ', s.recv(buffer_size))
        elif (msg == "goodbye"):
            print('\nSend goodbye msg.')
            len_msg = "%s\t%d\t" % (str(int(MessageType.Goodbye)), 123)
            s.send(len_msg.encode('utf-8'))
            time.sleep(.100)
            print('Received: ', s.recv(buffer_size))
        elif (msg == "kgs_test"):
            # Send meta data
            print('\nSend test data sample.')

            for i_sample in np.arange(0, input_test.shape[0]):
                data_sample = input_test[i_sample,:]
                data_sample_str = np.array2string(data_sample, threshold=np.inf, max_line_width=np.inf, separator=',')
                data_sample_str = data_sample_str.replace('[','')
                data_sample_str = data_sample_str.replace(']','')
                data_sample_msg = "%s\t%s\t" % (str(int(MessageType.DataSample)), data_sample_str)
                #print(data_sample_msg)
                s.send(data_sample_msg.encode('utf-8'))
                #time.sleep(.010)

                prediction_msg = s.recv(buffer_size)
                if prediction_msg:
                    split_data = prediction_msg.decode().split('\t')
                    if len(split_data) >= 1:
                        message_type = int(split_data[0])
                        if message_type == MessageType.PredictedClass:
                            split_payload = split_data[1].split(':')
                            if int(split_payload[0]) != -1:
                                print('SPOTTED: {} {}'.format(split_payload[0],split_payload[1]))

finally:    
    # Clean up the connection
    print('close connection')
    s.close()
