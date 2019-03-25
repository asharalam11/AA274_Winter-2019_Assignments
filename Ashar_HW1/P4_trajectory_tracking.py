import numpy as np
from numpy import linalg
from P3_pose_stabilization import ctrl_pose

def ctrl_traj(x, y, th,
              ctrl_prev,
              x_d, y_d,
              xd_d, yd_d,
              xdd_d, ydd_d,
              x_g, y_g, th_g):
    '''
    This function computes the closed-loop control law.
    Inputs:
        (x,y,th): current state
        ctrl_prev: previous control input (V,om)
        (x_d, y_d): desired position
        (xd_d, yd_d): desired velocity
        (xdd_d, ydd_d): desired acceleration
        (x_g,y_g,th_g): desired final state
    Outputs:
        (V, om): a numpy array np.array([V, om]) containing the desired control inputs
    '''

    # Timestep
    dt = 0.005
    
    ########## Code starts here ##########
    # Let's set gains
    kpx = 1.5
    kpy = 1.0
    kdx = 2.5
    kdy = 2.0

    rho = np.sqrt((x_g-x)**2 + (y_g-y)**2)   # Distance between current position and goal position

    # If the robot is sufficiently close (rho <= 0.5); switch to pose stabilisation controller from Problem 3
    if rho <= 0.5:
    	V, om = ctrl_pose(x, y, th, x_g, y_g, th_g)
    else: 	# Dealing with singularity
    	if abs(ctrl_prev[0]) <= 1e-6:
    		V_ = np.sign(ctrl_prev[0])* 1e-6
    	else:
    		V_ = ctrl_prev[0]

    	xd = V_ * np.cos(th)
    	yd = V_ * np.sin(th)

    	#Adding control inputs
        # As shown in the equation in Problem #2
    	u = np.array([xdd_d + kpx*(x_d-x) + kdx*(xd_d-xd), ydd_d + kpy*(y_d-y) + kdy*(yd_d-yd)])
    	J = np.array([[np.cos(th), -V_*np.sin(th)],[np.sin(th), V_*np.cos(th)]])

    	a, om = np.linalg.solve(J,u)
    	V = V_prev + a*dt

    # Keeping V < 0.5 and om < 1
    V_max = 0.5
    om_max = 1
    om = np.sign(om)*min(om_max,np.abs(om))
    V = np.sign(V)*min(V_max, np.abs(V))

    ########## Code ends here ##########

    return np.array([V, om])