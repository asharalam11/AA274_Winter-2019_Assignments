from sys import argv
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import argparse
from utils import car_dyn
from P3_pose_stabilization import ctrl_pose
from utils import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--parking',   type=str, default='forward',     help="parking type: forward, reverse, or parallel")
args = parser.parse_args()

'''
Fill the initial pose (x_0, y_0, th_0), and time duration (t_end) for forward, reverse, and parallel parking.
e.g.
x_0 = 5
y_0 = 3
th_0 = 0
t_end = 20
'''

if args.parking == "forward":
    ########## Code starts here ##########
	x_0 = 5
	y_0 = 3
	th_0 = np.pi/2
	t_end = 20
    ########## Code ends here ##########
elif args.parking == "reverse":
    ########## Code starts here ##########
	x_0 = 5
	y_0 = 3
	th_0 = -np.pi/2
	t_end = 20
    ########## Code ends here ##########
elif args.parking == "parallel":
    ########## Code starts here ##########
	x_0 = 7
	y_0 = 3
	th_0 = np.pi/2
	t_end = 20
    ########## Code ends here ##########
else:
    raise ValueError("parking type not known")


print '(x_0, y_0, th_0) = (%.2f, %.2f, %.2f)' %(x_0, y_0, th_0)
print 't_f = %.2f' %t_end# desired end pose
x_g = 5
y_g = 5
th_g = -np.pi/2

#timestep
dt = 0.005
N = int (t_end/dt)

# Set up simulation

time = dt * np.array(range(N+1)) 
state = np.zeros((N+1,3))
state[0,:] = np.array([[x_0, y_0, th_0]])
x = state[0,:]

ctrl = np.zeros((N,2))

for i in range(N): 
    ctrl_fbck = ctrl_pose(x[0], x[1], x[2], x_g, y_g, th_g)
    ctrl[i,0] = ctrl_fbck[0]
    ctrl[i,1] = ctrl_fbck[1]

    d_state = odeint(car_dyn, x, np.array([time[i], time[i+1]]), args=(ctrl[i,:], [0,0]) )
    x = d_state[1,:]
    state[i+1,:] = x

# Plots
maybe_makedirs('plots')
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(state[:,0],state[:,1],linewidth=1)
plt.title('Trajectory')
plt.quiver(state[0:-1:200,0],state[0:-1:200,1],np.cos(state[0:-1:200,2]), np.sin(state[0:-1:200,2]))
plt.grid('on')
plt.plot(x_0,y_0,'go',markerfacecolor='green',markersize=15)
plt.plot(x_g,y_g,'ro',markerfacecolor='red', markersize=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([3,8,2,6])
plt.tight_layout()


plt.subplot(1,3,2)
plt.plot(time, state[:,2],linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['$\\theta$ [rad]'],loc='best')
plt.tight_layout()


plt.subplot(1,3,3)
plt.plot(time[0:-1], ctrl, linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
plt.title("Control inputs")
plt.tight_layout()
plt.savefig("plots/" + args.parking + "_parking.png")
plt.show()
