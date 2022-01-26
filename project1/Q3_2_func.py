import numpy as np
from scipy.integrate import odeint

def func(theta_at_0): # Parameters are the initial values of θ
    m1 = 1
    m2 = 1
    L1 = 0.5
    L2 = 0.5
    g = 9.81

    def ode(x,t,m1,m2,L1,L2,g):
        x1,x2,x3,x4 = x
        tau = np.array([[0], [0]])
        M = np.mat([[m1*(L1**2) + m2*(L1**2 + 2*L1*L2*np.cos(x2) + L2**2), m2*(L1*L2*np.cos(x2) + L2**2)], [m2*(L1*L2*np.cos(x2) + L2**2), m2*(L2**2)]]) # M
        M_inv = np.mat([[1 / ((L1**2)*(m1 + m2*(np.sin(x2))**2)), -1*(L1*np.cos(x2) + L2) / ((L1**2)*L2*(m1 + m2*(np.sin(x2))**2))], \
                         [-1*(L1*np.cos(x2) + L2) / ((L1**2)*L2*(m1 + m2*(np.sin(x2))**2)), (m1*L1**2 + m2*(L1**2 + 2*L1*L2*np.cos(x2) + L2**2)) / (m2*(L1**2)*(L2**2)*(m1 + m2*(np.sin(x2))**2))]]) # Inverse of M
        C = np.array([[-1*m2*L1*L2*np.sin(x2)*(2*x3*x4 + x4**2)], [m2*L1*L2*(x3**2)*np.sin(x2)]]) # c
        G = np.array([[(m1+m2)*L1*g*np.cos(x1) + m2*g*L2*np.cos(x1 + x2)], [m2*g*L2*np.cos(x1 + x2)]]) # g(θ)
        x1_dot = x3
        x2_dot = x4
        x34_dot = np.dot(M.I, (tau - C - G))
        x3_dot = x34_dot[0,0]
        x4_dot = x34_dot[1,0]
        return [x1_dot, x2_dot, x3_dot, x4_dot]
    
    t_length = 5 # The time length is 5s
    dt = 0.001 # The time interval is 0.001s
    times = np.arange(0,t_length,dt)
    x_init = np.hstack((theta_at_0, np.array([0,0]))) # Angular speed is 0 when starting
    x_traj = odeint(ode, x_init, times, args=(m1,m2,L1,L2,g)) # Solve differential equations
    
    theta1 = x_traj[:,0] # θ1
    theta2 = x_traj[:,1] # θ2
    x1 = L1*np.cos(theta1) # x of link1 end
    y1 = L1*np.sin(theta1) # y of link1 end
    x2 = L1*np.cos(theta1) + L2*np.cos(theta1+theta2) # x of link2 end
    y2 = L1*np.sin(theta1) + L2*np.sin(theta1+theta2) # y of link2 end
    return [x1, y1, x2, y2]

input = np.array([0,np.pi/2]) # Set the initial values of θ
[x1, y1, x2, y2] = func(input) # Use the function





