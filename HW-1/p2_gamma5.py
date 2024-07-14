# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 01:12:19 2023

@author: sahil
"""
import numpy as np
import matplotlib.pyplot as plt

# With gamma = 5
G = np.array([[1, 0.2, 0.1],[0.1, 2, 0.1],[0.3, 0.1, 3]])
gamma = 5
alpha = 1.2
sigma = 0.1**2

t = np.arange(0,12,1)

# Second initial condition
p_init = np.array([[0.1],[0.01],[0.02]])

A = alpha*gamma*np.array([[0, G[0][1]/G[0][0], G[0][2]/G[0][0]],
                           [G[1][0]/G[1][1], 0, G[1][2]/G[1][1]],
                           [G[2][0]/G[2][2], G[2][1]/G[2][2], 0]])

B = alpha*gamma*np.array([[1/G[0][0]],
                          [1/G[1][1]],
                          [1/G[2][2]]])


P1arr = []
P2arr = []
P3arr = []
S1arr = []
S2arr = []
S3arr = []


p=p_init

for x in range(0,12):
    q1 = sigma + (0.2*p[1]) + (0.1*p[2])
    q2 = sigma + (0.1*p[0]) + (0.1*p[2])
    q3 = sigma + (0.3*p[0]) + (0.1*p[1])
    S = np.array([G[0][0]*p[0]/q1, G[1][1]*p[1]/q2, G[2][2]*p[2]/q3])
    P1arr.append(p[0])
    P2arr.append(p[1])
    P3arr.append(p[2])
    S1arr.append(S[0])
    S2arr.append(S[1])
    S3arr.append(S[2])
    p = np.dot(A,p) + np.dot(B,sigma)
    
target = [alpha*gamma]*12

plt.plot(t,S1arr, color='b', label='S1')
plt.plot(t,S2arr, color='orange', label='S2')
plt.plot(t,S3arr, color='g', label='S3')
plt.plot(t,target, color='r', label='Threshold')
plt.xlabel("t")
plt.ylabel("S")
plt.title("t x S (Second initial condition & gamma=5)")
plt.legend()
plt.show()

plt.plot(t,P1arr, color='b', label='P1')
plt.plot(t,P2arr, color='orange', label='P2')
plt.plot(t,P3arr, color='g', label='P3')


plt.xlabel("t")
plt.ylabel("P")
plt.title("t x P (Second initial condition & gamma=5)")
plt.legend()
plt.show()