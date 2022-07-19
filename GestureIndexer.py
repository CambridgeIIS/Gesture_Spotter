import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn import preprocessing
from data_utils import *

def plotDataBuffer(filename):

    data_buffer = np.loadtxt(filename, delimiter=',')

    data_buffer = preprocess(data_buffer)

    print(np.mean([data_buffer[i][0] for i in range(len(data_buffer))]))
    print(np.mean([data_buffer[i][1] for i in range(len(data_buffer))]))
    print(np.mean([data_buffer[i][2] for i in range(len(data_buffer))]))

    thumb_length_list = []
    index_length_list = []
    middle_length_list = []
    ring_length_list = []

    for i in range(len(data_buffer)):
        thumb_length = np.sqrt(
            np.power(data_buffer[i][3], 2) + np.power(data_buffer[i][4], 2) + np.power(data_buffer[i][5], 2))
        index_length = np.sqrt(
            np.power(data_buffer[i][6], 2) + np.power(data_buffer[i][7], 2) + np.power(data_buffer[i][8], 2))

        middle_length = np.sqrt(
            np.power(data_buffer[i][9], 2) + np.power(data_buffer[i][10], 2) + np.power(data_buffer[i][11], 2))
        ring_length = np.sqrt(
            np.power(data_buffer[i][12], 2) + np.power(data_buffer[i][13], 2) + np.power(data_buffer[i][14], 2))

        thumb_length_list.append(thumb_length)
        index_length_list.append(index_length)
        middle_length_list.append(middle_length)
        ring_length_list.append(ring_length)

    print('thumb: {}'.format(np.mean(thumb_length_list)))
    print('index: {}'.format(np.mean(index_length_list)))
    print('middle: {}'.format(np.mean(middle_length_list)))
    print('ring: {}'.format(np.mean(ring_length_list)))


    # print(data_buffer)

    # Define initial parameters
    init_frequency = 0

    # Create the figure and the lines that we will manipulate    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches    
    fig = plt.figure(figsize=(800*px, 900*px))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    idx = 0
    f_thumb, = ax1.plot([data_buffer[idx,0], data_buffer[idx,3]], [data_buffer[idx,1], data_buffer[idx,4]], [data_buffer[idx,2], data_buffer[idx,5]], 'o-', lw=2)
    f_index, = ax1.plot([data_buffer[idx,0], data_buffer[idx,6]], [data_buffer[idx,1], data_buffer[idx,7]], [data_buffer[idx,2], data_buffer[idx,8]], 'o-', lw=2)
    f_middle, = ax1.plot([data_buffer[idx,0], data_buffer[idx,9]], [data_buffer[idx,1], data_buffer[idx,10]], [data_buffer[idx,2], data_buffer[idx,11]], 'o-', lw=2)
    f_ring, = ax1.plot([data_buffer[idx,0], data_buffer[idx,12]], [data_buffer[idx,1], data_buffer[idx,13]], [data_buffer[idx,2], data_buffer[idx,14]], 'o-', lw=2)
    f_pinky, = ax1.plot([data_buffer[idx,0], data_buffer[idx,15]], [data_buffer[idx,1], data_buffer[idx,16]], [data_buffer[idx,2], data_buffer[idx,17]], 'o-', lw=2)

    # Axes
    scale = 0.5
    ax1.set_xlim([scale*0, scale*2])
    ax1.set_xlabel('x-axis')
    ax1.set_ylim([2.5+scale*-1, 2.5+scale*1])
    ax1.set_ylabel('y-axis')
    ax1.set_zlim([scale*-1, scale*1])
    ax1.set_zlabel('z-axis')
    ax1.view_init(elev=45, azim=-45)

    axcolor = 'lightgoldenrodyellow'

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Make a horizontal slider to control the sample index.
    axsample = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sample_idx_slider = Slider(
        ax=axsample,
        label='Sample Index',
        valmin=0,
        valmax=data_buffer.shape[0]-1,
        valinit=init_frequency,
        valfmt="%i",
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        idx = int(val)
        f_thumb.set_data_3d([data_buffer[idx,0], data_buffer[idx,3]], [data_buffer[idx,1], data_buffer[idx,4]], [data_buffer[idx,2], data_buffer[idx,5]])
        f_index.set_data_3d([data_buffer[idx,0], data_buffer[idx,6]], [data_buffer[idx,1], data_buffer[idx,7]], [data_buffer[idx,2], data_buffer[idx,8]])
        f_middle.set_data_3d([data_buffer[idx,0], data_buffer[idx,9]], [data_buffer[idx,1], data_buffer[idx,10]], [data_buffer[idx,2], data_buffer[idx,11]])
        f_ring.set_data_3d([data_buffer[idx,0], data_buffer[idx,12]], [data_buffer[idx,1], data_buffer[idx,13]], [data_buffer[idx,2], data_buffer[idx,14]])
        f_pinky.set_data_3d([data_buffer[idx,0], data_buffer[idx,15]], [data_buffer[idx,1], data_buffer[idx,16]], [data_buffer[idx,2], data_buffer[idx,17]])
        fig.canvas.draw_idle()


    # register the update function with each slider
    sample_idx_slider.on_changed(update)


    plt.show()


# plotDataBuffer('data/test_shrec.csv')

plotDataBuffer('logs/data_buffer_20220223_174805.csv')