from sys import argv
import argparse
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from utils import *
from P4_trajectory_tracking import ctrl_traj

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--x0',   type=float, default=0,         help="initial x")
parser.add_argument('--y0',   type=float, default=0,         help="initial y")
parser.add_argument('--th0',   type=float, default=-1.57,     help="initial th")
parser.add_argument('--dist',   type=int, default=0,         help="disturbance flag")
parser.add_argument('--ctrl',   type=str, default='closed',     help="open-loop or closed-loop control")
parser.add_argument('--data',   type=str, default='differential_flatness',     help="method for nominal traj.")
args = parser.parse_args()

# unpack argv
# script_name, filename, x_0, y_0, th_0, dist, ctrl = argv
x_0 = args.x0
y_0 = args.y0
th_0 = args.th0
dist = args.dist
ctrl = args.ctrl
filename = 'data/' + args.data + '.pkl'

model_name = 'sim_traj_' + args.data + '_' + str(args.dist) + '_' + args.ctrl

print 'filename: %s' % filename
print '(x_0, y_0, th_0) = (%.2f, %.2f, %.2f)' %(x_0, y_0, th_0)
print 'Noise: %i' %dist
print 'Control: %s' %ctrl

data_dict = load_dict(filename)
data = data_dict['z']
V = data_dict['V']
om = data_dict['om']

x_g = data[-1,0]
y_g = data[-1,1]
th_g = data[-1,2]

N = [len(data)-1]
dt = 0.005
t_end = N[0]*dt

noise = np.zeros((N[0]+1,2))
w_noise = 0
n_runs = 1
feedback = False

if dist or ctrl == 'closed': n_runs = 2

if dist: w_noise = 1

if ctrl == 'closed': feedback = True

# Setup Simulation
time = [dt * np.array(range(N[0] + 1))] 
state = [np.zeros((N[0] + 1, 3))]
state[0][0,:] = data[0, 0:3]
ctrl = [np.zeros((N[0], 2))]
if n_runs == 2:
    if feedback: t_end_2 = 1.2 * t_end
    else: t_end_2 = t_end
    N.append(int(t_end_2/dt))
    time.append(dt * np.array(range(N[1] + 1)))

    state.append(np.zeros((N[1] + 1,3)))
    state[1][0,:] = np.array([[x_0, y_0, th_0]])
    ctrl.append(np.zeros((N[1], 2)))

# Simulate
for n in range(n_runs):
    x = state[n][0,:]
    ctrl_prev = np.array([V[0], om[0]])

    if n == 1:
        noise = w_noise*np.vstack([np.sqrt(0.1)*np.random.randn(N[1]), np.sqrt(0.1)*np.random.randn(N[1])])
        noise = noise.T

    for i in range(N[n]): #t[0]...t[N-1]
        idx = N[0] if (n==1) and (i >= N[0]) else i
        # if (n==1) and (i >= N[0]): idx = N[0]
        # else: idx = i

        if n == 0:
            #Open-loop
            ctrl[n][i,:] = np.array([V[i], om[i]])

        elif (n == 1) and (feedback==False):
            #Open-loop
            ctrl[n][i,:] = np.array([V[i], om[i]])
        else:
            #Closed-loop
            ctrl_fbck = ctrl_traj(x[0], x[1], x[2], 
                                 ctrl_prev, 
                                 data[idx,0], data[idx,1], 
                                 data[idx,3], data[idx,4],
                                 data[idx,5], data[idx,6],
                                 x_g, y_g, th_g)
            ctrl[n][i,:] = ctrl_fbck
            ctrl_prev = ctrl[n][i,:]

        d_state = odeint(car_dyn, x, np.array([time[n][i], time[n][i+1]]), args=(ctrl[n][i,:], noise[i,:]))
        x = d_state[1,:]
        state[n][i+1,:] = x

# Plots
maybe_makedirs('plots')
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)

for n in range(n_runs):
    plt.plot(state[n][:,0],state[n][:,1],linewidth=2)

if (n_runs == 2) and dist:
    plt.legend(['Without noise', 'With noise'], loc="best")
elif (n_runs == 2) and ~dist:
    plt.legend(['Open-loop', 'Closed-loop'], loc="best")

plt.grid('on')
plt.plot(x_0, y_0, 'go', markerfacecolor='green', markersize=15)
plt.plot(x_g, y_g, 'ro', markerfacecolor='red', markersize=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-1, 6, -1, 6])
plt.title('Path')

plt.subplot(1,3,2)
plt.plot(time[0][0:-1], ctrl[0],linewidth=2)
plt.grid('on')
plt.xlabel('Time [s]')
plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
plt.axis([0, 21, -1.2, 1.2])
plt.tight_layout()
if dist: 
    plt.title('Without noise')
else: 
    plt.title('Open-loop')

if n_runs == 2:
    plt.subplot(1, 3 ,3)
    plt.plot(time[1][0:-1], ctrl[1], linewidth=2)
    plt.grid('on')
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.axis([0, 25, -1.2, 1.2])
    plt.tight_layout()
    if dist: plt.title('With noise')
    else: plt.title('Closed-loop')
plt.savefig('plots/' + model_name + '.png')
plt.show()