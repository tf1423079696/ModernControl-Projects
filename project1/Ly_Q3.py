import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

L1 = 0.5
L2 = 0.5

def Mtheta(m1,m2,L1,L2,x2):
    M = np.mat([[m1*(L1**2) + m2*(L1**2 + 2*L1*L2*np.cos(x2) + L2**2), m2*(L1*L2*np.cos(x2) + L2**2)], [m2*(L1*L2*np.cos(x2) + L2**2), m2*(L2**2)]])
    return M

def Mtheta_inv(m1,m2,L1,L2,x2):
    M_inv = np.mat([[1 / ((L1**2)*(m1 + m2*(np.sin(x2))**2)), -1*(L1*np.cos(x2) + L2) / ((L1**2)*L2*(m1 + m2*(np.sin(x2))**2))], [-1*(L1*np.cos(x2) + L2) / ((L1**2)*L2*(m1 + m2*(np.sin(x2))**2)), (m1*L1**2 + m2*(L1**2 + 2*L1*L2*np.cos(x2) + L2**2)) / (m2*(L1**2)*(L2**2)*(m1 + m2*(np.sin(x2))**2))]])
    return M_inv

def Ctheta(m2,L1,L2,x2,x3,x4):
    C = np.mat([[-1*m2*L1*L2*np.sin(x2)*(2*x3*x4 + x4**2)],[m2*L1*L2*(x3**2)*np.sin(x2)]]) 
    return C

def Gtheta(m1,m2,L1,L2,g,x1,x2):
    G = np.mat([[(m1+m2)*L1*g*np.cos(x1) + m2*g*L2*np.cos(x1 + x2)],[m2*g*L2*np.cos(x1 + x2)]]) 
    return G

def func(theta_at_0):
    m1 = 1
    m2 = 1
    L1 = 0.5
    L2 = 0.5
    g = 9.81
    def ode(x,t,m1,m2,L1,L2,g):
        x1,x2,x3,x4 = x
        tau = np.mat([[0], [0]])
        M = Mtheta(m1,m2,L1,L2,x2)
        M_inv = Mtheta_inv(m1,m2,L1,L2,x2)
        C = Ctheta(m2,L1,L2,x2,x3,x4)
        G = Gtheta(m1,m2,L1,L2,g,x1,x2)
        x1_dot = x3
        x2_dot = x4
        x34_dot = np.dot(M.I, (tau - C - G))
        x3_dot = x34_dot[0,0]
        x4_dot = x34_dot[1,0]
        return [x1_dot, x2_dot, x3_dot, x4_dot]
    
    t_length = 5 
    dt = 0.001 
    times = np.arange(0,t_length,dt)
    x_init = np.hstack((theta_at_0, np.array([0,0])))
    x_traj = odeint(ode, x_init, times, args=(m1,m2,L1,L2,g)) 
    return x_traj
    
input = np.array([0,np.pi/2]) 
line = func(input)
theta1 = line[:,0]
theta2 = line[:,1]
x1 = L1*np.cos(theta1)
y1 = L1*np.sin(theta1)
x2 = L1*np.cos(theta1) + L2*np.cos(theta1+theta2) 
y2 = L1*np.sin(theta1) + L2*np.sin(theta1+theta2)
p1 = [x1, y1] 
p2 = [x2, y2]