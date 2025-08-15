import numpy as np
import matplotlib.pyplot as plt

import random

import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
# %matplotlib inline

#QUESTION 2.a #QUESTION 2.a #QUESTION 2.a #QUESTION 2.a
#QUESTION 2.a #QUESTION 2.a #QUESTION 2.a #QUESTION 2.a
#QUESTION 2.a #QUESTION 2.a #QUESTION 2.a #QUESTION 2.a


x = np.arange(0, 1, 0.001)
y=[]
for i in x:
 a = 1.5*i + 5
 y.append(a)
plt.plot(x,y)

#QUESTION 2.b #QUESTION 2.b #QUESTION 2.b #QUESTION 2.b
#QUESTION 2.b #QUESTION 2.b #QUESTION 2.b #QUESTION 2.b
#QUESTION 2.b #QUESTION 2.b #QUESTION 2.b #QUESTION 2.b


x = np.arange(0, 1, 0.001)
y=[]
for i in x:
   a = 1.5*i + 5
   y.append(a)
r = []
for j in range(0, 1000):
   b=random.uniform(0, 1)
   r.append(b)
y1 =[]
for k in range(0, 1000):
   c=y[k] +(r[k]*0.1)
   y1.append(c)
  
plt.scatter(x, y1)


#QUESTION 2.c #QUESTION 2.c #QUESTION 2.c #QUESTION 2.c
#QUESTION 2.c #QUESTION 2.c #QUESTION 2.c #QUESTION 2.c
#QUESTION 2.c #QUESTION 2.c #QUESTION 2.c #QUESTION 2.c


sns.set()
 
x = np.arange(0, 1, 0.001)
y=[]
for i in x:
   a = 1*i + 10
   y.append(a)
r = []
for j in range(0, 1000):
   b=random.uniform(0, 1)
   r.append(b)
y1 =[]
for k in range(0, 1000):
   c=y[k] +(r[k]*0.1)
   y1.append(c)
def J(w0, w1, x, y1):
   J = 0
   for i in range(1000):
       J += ((w0 + w1*x[i]) - y1[i] )**2
   return J/2000 
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
w0 = np.linspace(-10,10,1000)
w1 = np.linspace(-10,10,1000)
aa0, aa1 = np.meshgrid(w0, w1)
ax.plot_surface(aa0, aa1, J(aa0,aa1,x,y1), rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('w1')
ax.set_ylabel('w0')
ax.set_zlabel('error')
 
plt.show()
 
 


#QUESTION 2.d #QUESTION 2.d #QUESTION 2.d #QUESTION 2.d
#QUESTION 2.d #QUESTION 2.d #QUESTION 2.d #QUESTION 2.d
#QUESTION 2.d #QUESTION 2.d #QUESTION 2.d #QUESTION 2.d

# Gradient descent
w1_init = -7 # initialization 
w0_init = -5
lr = 0.6  # learning rate (0.9 diverges, 0.6 quite interesting)
eps = 0.000001

# write your code here

w1_init = -7 # initialization 
w0_init = -5
lr = 0.6  # learning rate (0.9 diverges, 0.6 quite interesting)
eps = 0.000001
w = np.array([[-7,-5]])
indexs = [0,1]
num=100
x = x.reshape(num,1)
y_cor = y_cor.reshape(num,1)
print(x.shape)
w,w11,e1,epoch = gradient_descent(w,eps,x,y_cor,lr,indexs)
print(w11.shape)
print(w)

w11 = w11.reshape(epoch,2)
print(w11[:,1].reshape(epoch,1).shape)
# w0_, w1_ = np.meshgrid(w11[:,0].reshape(76,1), w11[:,1].reshape(76,1))


# ax = plt.axes(projection='3d')
# ax.plot_surface(w0_,w1_, error)
# plt.show()
# e1 = e1.reshape(76,1)
# plt.contour(w11[:,0].reshape(76,1),w11[:,1].reshape(76,1),e1)
# plt.show()
plt.contour(w0,w1,error,levels = 50)
w0_gd = w11[:,0].reshape(epoch,1)
w1_gd = w11[:,1].reshape(epoch,1)
# print(w0_gd)
plt.plot(w1_gd,w0_gd,'r')
# y_pred = w[0] + w[1]*x
plt.show()
y_pred = w[0,0] + w[0,1]*x
plt.scatter(x,y_cor)
plt.plot(x,y_pred,'r')
plt.show()