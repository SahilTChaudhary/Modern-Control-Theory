import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Exercise 5 (c)
def originalSys(X):
    x1,x2 = X
    return [x2 - x1*(x2**2), -x1**3]

def linearizedSys(X):
    x1,x2 = X
    return [x2,0]

x1 = np.linspace(-4,4,20)
x2 = np.linspace(-4,4,20)
X1,X2 = np.meshgrid(x1,x2)
u1,v1 = np.zeros(X1.shape),np.zeros(X2.shape)
u2,v2 = np.zeros(X1.shape),np.zeros(X2.shape)
A,B = X1.shape

for i in range(A):
    for j in range(B):
        x = X1[i,j]
        y = X2[i,j]
        x_dot = originalSys([x,y])
        xlin_dot = linearizedSys([x,y])
        u1[i,j] = x_dot[0]
        v1[i,j] = x_dot[1]
        u2[i,j] = xlin_dot[0]
        v2[i,j] = xlin_dot[1]
        
plot1 = plt.quiver(X1, X2, u1, v1, color='r')
plt.title("Original Non-Linear System")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()

plot2 = plt.quiver(X1, X2, u2, v2, color='g')
plt.title("Linearized System")
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()

# Exercise 5 (d)
xd1 = np.linspace(-1,1,20)
xd2 = np.linspace(-1,1,20)

XD1,XD2 = np.meshgrid(xd1,xd2)

V_dot = -4*(XD1**4) * (XD2**2)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XD1,XD2,V_dot,cmap = 'plasma')
ax.set_xlabel('x1')
ax.set_xlabel('x2')
plt.title("Variation of V_dot")
plt.show()