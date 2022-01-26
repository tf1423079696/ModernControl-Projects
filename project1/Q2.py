import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize, least_squares

c = 0.3
xi = np.array([-20000, 1000, -3000, 4000, -3500, 1000, 4000, -4000, 6000])
yi = np.array([5000, 3500, -4000, 1000, 2000, 6000, -3000, -1500, -1000])
ti = np.array([79445, 20009, 21622, 13683, 24709, 28223, 11293, 22990, 16446])

def func(theta,x,y):
    x1,x2,tau = theta # Define θ
    return 1/c * np.sqrt((x-x1)**2+(y-x2)**2) + tau # Define g(θ)

def error(theta,x,y,t): # Define t-g(θ)
    return t-func(theta,x,y)

def LScost(theta, x, y, t):
    x1,x2,tau = theta
    J = 0
    for i in range(np.size(t, 0)):
        J+= np.square(t[i] - (1/c * np.sqrt(np.square(x[i]-x1)+np.square(y[i]-x2)) + tau)) # Define the total cost       
    return J.item(0)

theta0=[rand.randint(-3000,3000),rand.randint(-3000,3000),\
        rand.randint(-5000,5000)] # Set the initial value of θ randomly
#para = leastsq(error,theta0,args=(xi,yi,ti)) # Solve the nonlinear least square problem with "leastsq"
#x1,x2,tau = para[0] # Get the value of θ
#para = minimize(LScost,theta0,args=(xi,yi,ti),method='SLSQP') # Solve the nonlinear least square problem with "minimize"
para = least_squares(error,theta0,args=(xi,yi,ti))
print(para)
x1, x2, tau = para.x

print("x1_init = ",theta0[0]," x2_init = ",theta0[1]," τ_init = ",theta0[2])
print("x1 = ",x1," x2 = ",x2," τ = ",tau)