import numpy as np
from scipy.integrate import odeint
from scipy.optimize import leastsq, minimize
import random

m1 = 1
m2 = 1
L1 = 0.5
L2 = 0.5
g = 9.81

def func(x,t,m1,m2,L1,L2,g):
    x1,x2,x3,x4 = x
    tau = np.array([[0], [0]])
    M = np.mat([[m1*(L1**2) + m2*(L1**2 + 2*L1*L2*np.cos(x2) + L2**2), m2*(L1*L2*np.cos(x2) + L2**2)], [m2*(L1*L2*np.cos(x2) + L2**2), m2*(L2**2)]])
    M_inv = np.mat([[1 / ((L1**2)*(m1 + m2*(np.sin(x2))**2)), -1*(L1*np.cos(x2) + L2) / ((L1**2)*L2*(m1 + m2*(np.sin(x2))**2))], [-1*(L1*np.cos(x2) + L2) / ((L1**2)*L2*(m1 + m2*(np.sin(x2))**2)), (m1*L1**2 + m2*(L1**2 + 2*L1*L2*np.cos(x2) + L2**2)) / (m2*(L1**2)*(L2**2)*(m1 + m2*(np.sin(x2))**2))]])
    C = np.array([[-1*m2*L1*L2*np.sin(x2)*(2*x3*x4 + x4**2)], [m2*L1*L2*(x3**2)*np.sin(x2)]])
    G = np.array([[(m1+m2)*L1*g*np.cos(x1) + m2*g*L2*np.cos(x1 + x2)], [m2*g*L2*np.cos(x1 + x2)]])
    x1_dot = x3
    x2_dot = x4
    x34_dot = np.dot(M.I, (tau - C - G))
    x3_dot = x34_dot[0,0]
    x4_dot = x34_dot[1,0]
    return [x1_dot, x2_dot, x3_dot, x4_dot]

input = np.array([0,np.pi/2])
t_range = 20
dt = 0.001
times = np.arange(0,t_range,dt)
x_init = np.hstack((input,np.array([0,0])))

g = 9.81
x_traj = odeint(func, x_init, times, args=(m1,m2,L1,L2,g)) # Compute θ1 θ2 θ1_dot θ2_dot
length = np.size(x_traj,0)
theta1_dot_dot = np.zeros(length) 
theta2_dot_dot = np.zeros(length)
for i in range(length):
    M_pred = np.mat([[m1*(L1**2) + m2*(L1**2 + 2*L1*L2*np.cos(x_traj[i,1]) + L2**2), m2*(L1*L2*np.cos(x_traj[i,1]) + L2**2)], [m2*(L1*L2*np.cos(x_traj[i,1]) + L2**2), m2*(L2**2)]])
    C_pred = np.mat([[-1*m2*L1*L2*np.sin(x_traj[i,1])*(2*x_traj[i,2]*x_traj[i,3] + x_traj[i,3]**2)], [m2*L1*L2*(x_traj[i,2]**2)*np.sin(x_traj[i,1])]])
    G_pred = np.mat([[(m1+m2)*L1*g*np.cos(x_traj[i,0]) + m2*g*L2*np.cos(x_traj[i,0] + x_traj[i,1])], [m2*g*L2*np.cos(x_traj[i,0] + x_traj[i,1])]])
    tau = np.mat([[0], [0]])
    theta12_dot2 = np.dot(M_pred.I, (tau - C_pred - G_pred)) # Compute θ1_dot_dot and θ2_dot_dot
    theta1_dot_dot[i] = theta12_dot2[0,0] + random.uniform(-1,1) # Obtain measurement data of θ1_dot_dot by adding a noise
    theta2_dot_dot[i] = theta12_dot2[1,0] + random.uniform(-1,1) # Obtain measurement data of θ2_dot_dot by adding a noise

# g(θ) consists of theta1_dot2func and theta2_dot2func.
def theta1_dot2func(para, x1, x2, x3, x4, tau1, tau2):
    m1_pred, m2_pred, L1_pred, L2_pred = para
    g = 9.81
    M11_inv = 1 / ((L1_pred**2)*(m1_pred + m2_pred*(np.sin(x2))**2))
    M12_inv = -1*(L1_pred*np.cos(x2) + L2_pred) / ((L1_pred**2)*L2_pred*(m1_pred + m2_pred*(np.sin(x2))**2))
    C1 = -m2_pred*L1_pred*L2_pred*np.sin(x2)*(2*x3*x4 + x4**2)
    G1 = (m1_pred+m2_pred)*L1_pred*g*np.cos(x1) + m2_pred*g*L2_pred*np.cos(x1 + x2)
    C2 = m2_pred*L1_pred*L2_pred*(x3**2)*np.sin(x2)
    G2 = m2_pred*g*L2_pred*np.cos(x1 + x2)
    return M11_inv*(tau1 - C1 - G1) + M12_inv*(tau2 - C2 - G2)

def theta2_dot2func(para, x1, x2, x3, x4, tau1, tau2):
    m1_pred, m2_pred, L1_pred, L2_pred = para
    g = 9.81
    M21_inv = -1*(L1_pred*np.cos(x2) + L2_pred) / ((L1_pred**2)*L2_pred*(m1_pred + m2_pred*(np.sin(x2))**2))
    M22_inv = (m1_pred*L1_pred**2 + m2_pred*(L1_pred**2 + 2*L1_pred*L2_pred*np.cos(x2) + L2_pred**2)) / (m2_pred*(L1_pred**2)*(L2_pred**2)*(m1_pred + m2_pred*(np.sin(x2))**2))
    C1 = -m2_pred*L1_pred*L2_pred*np.sin(x2)*(2*x3*x4 + x4**2)
    G1 = (m1_pred+m2_pred)*L1_pred*g*np.cos(x1) + m2_pred*g*L2_pred*np.cos(x1 + x2)
    C2 = m2_pred*L1_pred*L2_pred*(x3**2)*np.sin(x2)
    G2 = m2_pred*g*L2_pred*np.cos(x1 + x2)
    return M21_inv*(tau1 - C1 - G1) + M22_inv*(tau2 - C2 - G2)

# error used by leastsq, while LScost used by minimize
def error(para, x1, x2, x3, x4, tau1, tau2, theta1_dot2, theta2_dot2):
    return np.sqrt((theta1_dot2 - theta1_dot2func(para, x1, x2, x3, x4, tau1, tau2))**2 + (theta2_dot2 - theta2_dot2func(para, x1, x2, x3, x4, tau1, tau2))**2)

tau1 = np.zeros(length) # τ1=τ2=0 when free falling
tau2 = np.zeros(length)
para0 = [9.5, 9.5, 2, 2] # set initial values of parameters when theta2_init = pi/2
#para0 = [4.5, 4.5, 2.5, 2] # set initial values of parameters when theta2_init = -pi/2
para = leastsq(error, para0, args=(x_traj[:length,0], x_traj[:length,1], x_traj[:length,2], x_traj[:length,3], tau1, tau2, theta1_dot_dot, theta2_dot_dot))
#para = least_squares(error, para0, args=(x_traj[:length,0], x_traj[:length,1], x_traj[:length,2], x_traj[:length,3], tau1, tau2, theta1_dot_dot, theta2_dot_dot))

m1_pred,m2_pred,L1_pred,L2_pred = para[0]
#print(para)
#m1_pred,m2_pred,L1_pred,L2_pred = para.x
print("theta2_init = ",input[1], " t_length = ",t_range)
print("m1 = ",m1_pred," m2 = ",m2_pred," L1 = ",L1_pred," L2 = ",L2_pred)