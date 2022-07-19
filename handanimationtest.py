from handanimation import HandAnimation
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')





up_data = np.load('data/test_SHRINK.npy')


gross_up = up_data[:, :3]
fine_up = up_data[:, 3:]

thumbs_data = np.load('data/test_GROW.npy')

gross_thumbs = thumbs_data[:, :3]
fine_thumbs = thumbs_data[:, 3:]

origin = np.array([0, 0, 0])

hand_animation_up = HandAnimation(origin, gross_up, fine_up)
time_up = 0

hand_animation_thumbs = HandAnimation(origin, gross_thumbs, fine_thumbs)
time_thumbs = 0

scale = 1

shift_x = 0.0
shift_y = 0.0
shift_z = -0.4
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')


ax1.set_xlim([0.1*scale+shift_x, 0.7*scale+shift_x])
ax1.set_ylim([-0.3*scale+shift_y, 0.2*scale+shift_y])
ax1.set_zlim([0*scale+shift_z, 0.3*scale+shift_z])
ax1.view_init(elev=-0, azim=-45)

finger1_up, = ax1.plot([], [], 'o-', lw=2)
finger2_up, = ax1.plot([], [], 'o-', lw=2)
finger3_up, = ax1.plot([], [], 'o-', lw=2)
finger4_up, = ax1.plot([], [], 'o-', lw=2)
finger5_up, = ax1.plot([], [], 'o-', lw=2)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_xlim([0.1*scale+shift_x, 0.7*scale+shift_x])
ax2.set_ylim([-0.3*scale+shift_y, 0.2*scale+shift_y])
ax2.set_zlim([0*scale+shift_z, 0.3*scale+shift_z])
ax2.view_init(elev=-0, azim=-45)

finger1_thumbs, = ax2.plot([], [], 'o-', lw=2)
finger2_thumbs, = ax2.plot([], [], 'o-', lw=2)
finger3_thumbs, = ax2.plot([], [], 'o-', lw=2)
finger4_thumbs, = ax2.plot([], [], 'o-', lw=2)
finger5_thumbs, = ax2.plot([], [], 'o-', lw=2)


class Index(object):
    right_preferred = True

    def right_clicked(self, event):
        self.right_preferred = True
        plt.close()
        print(self.right_preferred)

    def left_clicked(self, event):
        self.right_preferred = False
        plt.close()
        print(self.right_preferred)


callback = Index()
axleft = plt.axes([0.4, 0.05, 0.1, 0.075])
axright = plt.axes([0.51, 0.05, 0.1, 0.075])
bleft = Button(axleft, 'fake')
bleft.on_clicked(callback.left_clicked)
bright = Button(axright, 'real')
bright.on_clicked(callback.right_clicked)


def init():
    finger1_up.set_data([], [])
    finger2_up.set_data([], [])
    finger3_up.set_data([], [])
    finger4_up.set_data([], [])
    finger5_up.set_data([], [])

    finger1_thumbs.set_data([], [])
    finger2_thumbs.set_data([], [])
    finger3_thumbs.set_data([], [])
    finger4_thumbs.set_data([], [])
    finger5_thumbs.set_data([], [])
    return finger1_up, finger2_up, finger3_up, finger4_up, finger5_up, finger1_thumbs, finger2_thumbs, finger3_thumbs, finger4_thumbs, finger5_thumbs


def animate(i):
    global hand_animation_up, time_up, hand_animation_thumbs, time_thumbs
    hand_animation_up.step(time_up)
    time_up += 1

    hand_animation_thumbs.step(time_thumbs)
    time_thumbs += 1

    f1, f2, f3, f4, f5 = hand_animation_up.position()

    finger1_up.set_data(f1.T[0:2])
    finger1_up.set_3d_properties(f1.T[2])

    finger2_up.set_data(f2.T[0:2])
    finger2_up.set_3d_properties(f2.T[2])

    finger3_up.set_data(f3.T[0:2])
    finger3_up.set_3d_properties(f3.T[2])

    finger4_up.set_data(f4.T[0:2])
    finger4_up.set_3d_properties(f4.T[2])

    finger5_up.set_data(f5.T[0:2])
    finger5_up.set_3d_properties(f5.T[2])

    g1, g2, g3, g4, g5 = hand_animation_thumbs.position()

    finger1_thumbs.set_data(g1.T[0:2])
    finger1_thumbs.set_3d_properties(g1.T[2])

    finger2_thumbs.set_data(g2.T[0:2])
    finger2_thumbs.set_3d_properties(g2.T[2])

    finger3_thumbs.set_data(g3.T[0:2])
    finger3_thumbs.set_3d_properties(g3.T[2])

    finger4_thumbs.set_data(g4.T[0:2])
    finger4_thumbs.set_3d_properties(g4.T[2])

    finger5_thumbs.set_data(g5.T[0:2])
    finger5_thumbs.set_3d_properties(g5.T[2])

    if time_up >= hand_animation_up.max_time - 1 and time_thumbs >= hand_animation_thumbs.max_time - 1:
        time_up = 0
        time_thumbs = 0

    return finger1_up, finger2_up, finger3_up, finger4_up, finger5_up, finger1_thumbs, finger2_thumbs, finger3_thumbs, finger4_thumbs, finger5_thumbs



ani = animation.FuncAnimation(fig, animate, frames=200,
                              interval=20, blit=True, init_func=init, repeat=True)
writergif = animation.PillowWriter(fps=30)
ani.save('test0.gif',writer=writergif)

# ani.save('test0.mp4', writer="ffmpeg")
# plt.show()
