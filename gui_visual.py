import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import pandas as pd
from matplotlib.widgets import Slider

def visualize(data_file):

    data = pd.read_csv(data_file)
    print(np.shape(data))

    frame_num = data.shape[0]


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d([-0.5, 0.5])
    ax.set_ylim3d([-0.5, 0.5])
    ax.set_zlim3d([-0.5, 0.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Z') # WARNING: y and z axes swapped to match unity convention
    ax.set_zlabel('Y') # WARNING: y and z axes swapped to match unity convention

    #initialising all joints
    wrist_pos = ax.plot3D([data["wrs_x"][0]], [data["wrs_z"][0]], [data["wrs_y"][0]], marker="x")[0]

    pinky_0_1, = ax.plot3D([data["pi0_x"][0], data["pi1_x"][0]], [data["pi0_z"][0], data["pi1_z"][0]], [data["pi0_y"][0], data["pi1_y"][0]], 'yo-', ms=2)
    pinky_1_2, = ax.plot3D([data["pi1_x"][0], data["pi2_x"][0]], [data["pi1_z"][0], data["pi2_z"][0]], [data["pi1_y"][0], data["pi2_y"][0]], 'yo-', ms=2)
    pinky_2_3, = ax.plot3D([data["pi2_x"][0], data["pi3_x"][0]], [data["pi2_z"][0], data["pi3_z"][0]], [data["pi2_y"][0], data["pi3_y"][0]], 'yo-', ms=2)
    pinky_3_4, = ax.plot3D([data["pi3_x"][0], data["pi4_x"][0]], [data["pi3_z"][0], data["pi4_z"][0]], [data["pi3_y"][0], data["pi4_y"][0]], 'yo-', ms=2)

    backbone_3, = ax.plot3D([data["wrs_x"][0], data["ri1_x"][0]], [data["wrs_z"][0], data["ri1_z"][0]], [data["wrs_y"][0], data["ri1_y"][0]], 'co-', ms=2)
    ring_1_2, = ax.plot3D([data["ri1_x"][0], data["ri2_x"][0]], [data["ri1_z"][0], data["ri2_z"][0]], [data["ri1_y"][0], data["ri2_y"][0]], 'co-', ms=2)
    ring_2_3, = ax.plot3D([data["ri2_x"][0], data["ri3_x"][0]], [data["ri2_z"][0], data["ri3_z"][0]], [data["ri2_y"][0], data["ri3_y"][0]], 'co-', ms=2)
    ring_3_4, = ax.plot3D([data["ri3_x"][0], data["ri4_x"][0]], [data["ri3_z"][0], data["ri4_z"][0]], [data["ri3_y"][0], data["ri4_y"][0]], 'co-', ms=2)

    mid_1_2, = ax.plot3D([data["mi1_x"][0], data["mi2_x"][0]], [data["mi1_z"][0], data["mi2_z"][0]], [data["mi1_y"][0], data["mi2_y"][0]], 'ro-', ms=2)
    mid_2_3, = ax.plot3D([data["mi2_x"][0], data["mi3_x"][0]], [data["mi2_z"][0], data["mi3_z"][0]], [data["mi2_y"][0], data["mi3_y"][0]], 'ro-', ms=2)
    mid_3_4, = ax.plot3D([data["mi3_x"][0], data["mi4_x"][0]], [data["mi3_z"][0], data["mi4_z"][0]], [data["mi3_y"][0], data["mi4_y"][0]], 'ro-', ms=2)
    backbone_2, = ax.plot3D([data["wrs_x"][0], data["mi1_x"][0]], [data["wrs_z"][0], data["mi1_z"][0]], [data["wrs_y"][0], data["mi1_y"][0]], 'ro-', ms=2)

    index_1_2, = ax.plot3D([data["in1_x"][0], data["in2_x"][0]], [data["in1_z"][0], data["in2_z"][0]], [data["in1_y"][0], data["in2_y"][0]], 'go-', ms=2)
    index_2_3, = ax.plot3D([data["in2_x"][0], data["in3_x"][0]], [data["in2_z"][0], data["in3_z"][0]], [data["in2_y"][0], data["in3_y"][0]], 'go-', ms=2)
    index_3_4, = ax.plot3D([data["in3_x"][0], data["in4_x"][0]], [data["in3_z"][0], data["in4_z"][0]], [data["in3_y"][0], data["in4_y"][0]], 'go-', ms=2)
    backbone_1, = ax.plot3D([data["wrs_x"][0], data["in1_x"][0]], [data["wrs_z"][0], data["in1_z"][0]], [data["wrs_y"][0], data["in1_y"][0]], 'go-', ms=2)

    thumb_0_1, = ax.plot3D([data["th0_x"][0], data["th1_x"][0]], [data["th0_z"][0], data["th1_z"][0]], [data["th0_y"][0], data["th1_y"][0]], 'bo-', ms=2)
    thumb_1_2, = ax.plot3D([data["th1_x"][0], data["th2_x"][0]], [data["th1_z"][0], data["th2_z"][0]], [data["th1_y"][0], data["th2_y"][0]], 'bo-', ms=2)
    thumb_2_3, = ax.plot3D([data["th2_x"][0], data["th3_x"][0]], [data["th2_z"][0], data["th3_z"][0]], [data["th2_y"][0], data["th3_y"][0]], 'bo-', ms=2)
    thumb_3_4, = ax.plot3D([data["th3_x"][0], data["th4_x"][0]], [data["th3_z"][0], data["th4_z"][0]], [data["th3_y"][0], data["th4_y"][0]], 'bo-', ms=2)

    # Make a horizontal slider to control the frequency.
    axframe = plt.axes([0.2, 0.05, 0.7, 0.03])
    frame_slider = Slider(
        ax=axframe,
        label='Frame',
        valmin=0,
        valmax=frame_num,
        valinit=0,
        valstep=1)

    def update_lines(val):
        i = frame_slider.val
        wrist_pos.set_data_3d(data["wrs_x"][i], data["wrs_z"][i], data["wrs_y"][i])
        
        pinky_0_1.set_data_3d([data["pi0_x"][i], data["pi1_x"][i]], [data["pi0_z"][i], data["pi1_z"][i]], [data["pi0_y"][i], data["pi1_y"][i]])
        pinky_1_2.set_data_3d([data["pi1_x"][i], data["pi2_x"][i]], [data["pi1_z"][i], data["pi2_z"][i]], [data["pi1_y"][i], data["pi2_y"][i]])
        pinky_2_3.set_data_3d([data["pi2_x"][i], data["pi3_x"][i]], [data["pi2_z"][i], data["pi3_z"][i]], [data["pi2_y"][i], data["pi3_y"][i]])
        pinky_3_4.set_data_3d([data["pi3_x"][i], data["pi4_x"][i]], [data["pi3_z"][i], data["pi4_z"][i]], [data["pi3_y"][i], data["pi4_y"][i]])

        ring_1_2.set_data_3d([data["ri1_x"][i], data["ri2_x"][i]], [data["ri1_z"][i], data["ri2_z"][i]], [data["ri1_y"][i], data["ri2_y"][i]])
        ring_2_3.set_data_3d([data["ri2_x"][i], data["ri3_x"][i]], [data["ri2_z"][i], data["ri3_z"][i]], [data["ri2_y"][i], data["ri3_y"][i]])
        ring_3_4.set_data_3d([data["ri3_x"][i], data["ri4_x"][i]], [data["ri3_z"][i], data["ri4_z"][i]], [data["ri3_y"][i], data["ri4_y"][i]])
        backbone_3.set_data_3d([data["wrs_x"][i], data["ri1_x"][i]], [data["wrs_z"][i], data["ri1_z"][i]], [data["wrs_y"][i], data["ri1_y"][i]])
        
        mid_1_2.set_data_3d([data["mi1_x"][i], data["mi2_x"][i]], [data["mi1_z"][i], data["mi2_z"][i]], [data["mi1_y"][i], data["mi2_y"][i]])
        mid_2_3.set_data_3d([data["mi2_x"][i], data["mi3_x"][i]], [data["mi2_z"][i], data["mi3_z"][i]], [data["mi2_y"][i], data["mi3_y"][i]])
        mid_3_4.set_data_3d([data["mi3_x"][i], data["mi4_x"][i]], [data["mi3_z"][i], data["mi4_z"][i]], [data["mi3_y"][i], data["mi4_y"][i]])
        backbone_2.set_data_3d([data["wrs_x"][i], data["mi1_x"][i]], [data["wrs_z"][i], data["mi1_z"][i]], [data["wrs_y"][i], data["mi1_y"][i]])
        
        index_1_2.set_data_3d([data["in1_x"][i], data["in2_x"][i]], [data["in1_z"][i], data["in2_z"][i]], [data["in1_y"][i], data["in2_y"][i]])
        index_2_3.set_data_3d([data["in2_x"][i], data["in3_x"][i]], [data["in2_z"][i], data["in3_z"][i]], [data["in2_y"][i], data["in3_y"][i]])
        index_3_4.set_data_3d([data["in3_x"][i], data["in4_x"][i]], [data["in3_z"][i], data["in4_z"][i]], [data["in3_y"][i], data["in4_y"][i]])
        backbone_1.set_data_3d([data["wrs_x"][i], data["in1_x"][i]], [data["wrs_z"][i], data["in1_z"][i]], [data["wrs_y"][i], data["in1_y"][i]])
        
        thumb_0_1.set_data_3d([data["th0_x"][i], data["th1_x"][i]], [data["th0_z"][i], data["th1_z"][i]], [data["th0_y"][i], data["th1_y"][i]])
        thumb_1_2.set_data_3d([data["th1_x"][i], data["th2_x"][i]], [data["th1_z"][i], data["th2_z"][i]], [data["th1_y"][i], data["th2_y"][i]])
        thumb_2_3.set_data_3d([data["th2_x"][i], data["th3_x"][i]], [data["th2_z"][i], data["th3_z"][i]], [data["th2_y"][i], data["th3_y"][i]])
        thumb_3_4.set_data_3d([data["th3_x"][i], data["th4_x"][i]], [data["th3_z"][i], data["th4_z"][i]], [data["th3_y"][i], data["th4_y"][i]])

    frame_slider.on_changed(update_lines)
    plt.show()

    anim = FuncAnimation(fig, update_lines, 1500, interval=10, blit=True)
    plt.show()
