import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, linalg
import ipdb
import math
from control import StateSpace, poles

Ca, m, lf, lr, Iz = 20000, 1888.6, 1.55, 1.39, 25854

velocities = np.linspace(1,40,40)
singVals = []
pole1, pole2, pole3, pole4 = [], [], [], []

for xdot in velocities:
    A = np.array([[0, 1, 0, 0],
        [0, -4*Ca/(m*xdot), 4*Ca/m, -2*Ca*(lf-lr)/(m*xdot)],
        [0, 0, 0, 1],
        [0, -2*Ca*(lf-lr)/(Iz*xdot), 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2+lr**2)/(Iz*xdot)]])

    B = np.array([[0, 0],
        [2*Ca/m, 0],
        [0, 0],
        [2*Ca*lf/Iz, 0]])

    C = np.eye(4)
    D = np.zeros((4, 2))

    P = np.concatenate([B, A@B, A@A@B, A@A@A@B], axis=1)
    U,S,V = np.linalg.svd(P)

    singVals.append(math.log10(S[0]/S[3]))
    
    ss = StateSpace(A, B, C, D)
    pole = poles(ss)
    pole1.append(pole[0].real)
    pole2.append(pole[1].real)
    pole3.append(pole[2].real)
    pole4.append(pole[3].real)
    
plt.figure
plt.plot(velocities, singVals)
plt.title('log10(sigma_1/sigma_n) vs Velocities')
plt.xlabel('Velocities')
plt.ylabel('log10(sigma_1/sigma_n)')
plt.show()

plt.figure
plt.subplot(2,2,1)
plt.plot(velocities,pole1)
plt.title('Pole-1 vs Velocities')
plt.xlabel('Velocities')
plt.ylabel('Pole-1')

plt.subplot(2,2,2)
plt.plot(velocities,pole2)
plt.title('Pole-2 vs Velocities')
plt.xlabel('Velocities')
plt.ylabel('Pole-2')

plt.subplot(2,2,3)
plt.plot(velocities,pole3)
plt.title('Pole-3 vs Velocities')
plt.xlabel('Velocities')
plt.ylabel('Pole-3')

plt.subplot(2,2,4)
plt.plot(velocities,pole4)
plt.title('Pole-4 vs Velocities')
plt.xlabel('Velocities')
plt.ylabel('Pole-4')

plt.show()