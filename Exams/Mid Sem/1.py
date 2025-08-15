from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math as m
# from sympy import *

x = np.arange(-10,10)
y = np.arange(-10,10)
X,Y = np.meshgrid(x,y)
Z = X*X + Y*Y +2*X +2*Y


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

x = np.arange(-10, 10 )  
y = np.arange(-10, 10  )
X,Y = np.meshgrid(x,y)

Z = X*np.sin(X) + Y*np.sin(Y)
#f2 = np.vectorize(Z)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()