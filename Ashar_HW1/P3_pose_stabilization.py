import numpy as np
from utils import wrapToPi
from math import sqrt

def ctrl_pose(x, y, th, xg, yg, thg):
    '''
    This function implements the pose stabilization controller.
    Inputs:
        x, y, th: the current pose of the robot
        xg, yg, thg: the desired pose of the robot
    Outputs:
        ctrl: a numpy array np.array([V, om]) containing the desired control inputs
    HINT: you need to use the wrapToPi function
    HINT: don't forget to saturate your control inputs
    '''
    
    ########## Code starts here ##########
    V_maximum = 0.5
    omega = 1
    del_x = xg-x
    del_y = yg-y
    rho = np.sqrt((del_x**2) + (del_y**2))
    angle_compos = wrapToPi(np.arctan2(del_y,del_x))
    alpha = angle_compos-th
    delta = angle_compos - thg
    k1 = 0.5
    k2 = 0.8
    k3 = 0.8
    V = k1*rho*np.cos(alpha)
    om = k2*alpha + (k1*np.sin(alpha)*np.cos(alpha)*(alpha+k3*delta)/(np.sinc(alpha)))
    # Saturating 
    V = np.minimum(np.maximum(V,-V_maximum),V_maximum)
    omega = np.minimum(np.maximum(om,-omega),omega)
    ctrl = [V, om]

    return np.array([V, om])
    ########## Code ends here ##########
