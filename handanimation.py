from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.widgets import Button

class HandAnimation:
    def __init__(self, origin, gross_data, fine_data, num_fingers=5):
        self.origin = origin
        self.gross_data = gross_data
        self.fine_data = fine_data
        self.num_fingers = num_fingers

        self.max_time = gross_data.shape[0]

        self.time_elapsed = 0
        self.state = self.initial_state()

    def initial_state(self):
        hand_com = self.gross_data[0] - self.origin
        finger_data = self.fine_data[0] - np.repeat(self.origin, self.num_fingers)
        state = np.concatenate([hand_com, finger_data])
        return np.split(state, self.num_fingers+1)
    
    def step(self, i):
        if (i < self.max_time - 1):

            hand_com = self.gross_data[i] - self.origin
            finger_data = self.fine_data[i] - np.repeat(self.origin, self.num_fingers)
            state = np.concatenate([hand_com, finger_data])
            self.state = np.split(state, self.num_fingers+1)
    
    def position(self):
        finger_1 = np.array([self.state[0], self.state[1]])

        finger_2 = np.array([self.state[0], self.state[2]])

        finger_3 = np.array([self.state[0], self.state[3]])

        finger_4 = np.array([self.state[0], self.state[4]])

        finger_5 = np.array([self.state[0], self.state[5]])

        return finger_1, finger_2, finger_3, finger_4, finger_5
