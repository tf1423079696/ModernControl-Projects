import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean, size
from scipy.optimize import leastsq

# read the wind speed data
windspeed = np.loadtxt('winddata.csv', delimiter=",", skiprows=1, usecols=1)
#print(windspeed)

err = np.zeros(20) #err：总预测误差
pred = np.zeros(shape=(20,5000)) #pred：风速预测值
pred[:][:] = windspeed[0:5000]

N = np.linspace(1,20,20).astype(int)
for n in N:
    H_T = np.zeros(shape=(n,5000-n)) # H_T: H.T
    y = np.mat(windspeed[n:5000].reshape(5000-n,1))
    for i in range(n):
        H_T[i][:] = windspeed[n-i-1:5000-(i+1)]
    H = np.mat(H_T.T)

    theta = np.dot(np.dot(np.dot(H.T, H).I, H.T), y) #计算θ
    err[n-1] = sum(np.dot((y - np.dot(H, theta)).T, (y - np.dot(H, theta)))) #计算err

    pred[n-1][n:5000] = np.dot(H, theta).reshape(1, 5000-n)
    #print(para)

print(err)
plt.plot(N,err,label="Total Prediction Error",linewidth=2)
plt.legend(loc='upper right')
plt.xticks(np.arange(1,21,1),np.arange(1,21,1))
plt.xlabel('n')
plt.ylabel('err(n)')
plt.show()

plt.plot(np.arange(1,5001,1), windspeed[0:5000], label="Actual Wind Speed")
plt.plot(np.arange(err.argmin()+2,5001,1), pred[err.argmin()][err.argmin()+1:], label="Predicted Wind Speed", linestyle='--')
plt.xlabel('k')
plt.ylabel('wind speed (unit: 100m/s)')
plt.legend(loc='upper right')
plt.show()