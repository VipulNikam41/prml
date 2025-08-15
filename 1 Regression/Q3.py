import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import random

import numpy
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
# %matplotlib inline


#QUESTION 3.a  #QUESTION 3.a  #QUESTION 3.a  #QUESTION 3.a
#QUESTION 3.a  #QUESTION 3.a  #QUESTION 3.a  #QUESTION 3.a
#QUESTION 3.a  #QUESTION 3.a  #QUESTION 3.a  #QUESTION 3.a



x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)
w0=1
w1=1
w2=1
p=[]
X1,X2 = np.meshgrid(x1, x2)
y = 1 + X1 + X2
for i in x1:
 for j in x2:
   a = 1 + i*w1 + j*w2
   p.append(a)
errorp = []
for i in range(900) :
   errorp.append(p[i]+random.random()*0.1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, y, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X1, X2, errorp,'.')
plt.show()



#QUESTION 3.b  #QUESTION 3.b  #QUESTION 3.b  #QUESTION 3.b
#QUESTION 3.b  #QUESTION 3.b  #QUESTION 3.b  #QUESTION 3.b
#QUESTION 3.b  #QUESTION 3.b  #QUESTION 3.b  #QUESTION 3.b


sns.set()
x1 = np.linspace(-1, 1, 30)
x2 = np.linspace(-1, 1, 30)
w0=1
w1=1
w2=1
p=[]
 
 
for i in x1:
   for j in x2:
       a = 1 + i*w1 + j*w2
       p.append(a)
y1 = []
for i in range(900) :
   y1.append(p[i]+random.random()*0.1)
 
def J(w1, w2,x1,y1):
   J = 0
   p=[]
   for i in x1:
       for j in x2:
           a = 1 + i*w1 + j*w2
           p.append(a)
   for i in range(900):
       J += (p[i] - y1[i] )**2
   return J/900
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
w1 = np.linspace(-10,10,100)
w2 = np.linspace(-10,10,100)
aa0, aa1 = np.meshgrid(w1, w2)
ax.plot_surface(aa0, aa1, J(aa0,aa1,x1,y1), rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('j')
plt.show()
plt.contour(w1,w2,J(aa0,aa1,x1,y1),60)
w3=w1
w4=w2
plt.show()



#QUESTION 3.c  #QUESTION 3.c  #QUESTION 3.c  #QUESTION 3.c
#QUESTION 3.c  #QUESTION 3.c  #QUESTION 3.c  #QUESTION 3.c
#QUESTION 3.c  #QUESTION 3.c  #QUESTION 3.c  #QUESTION 3.c

y = w0 + 1*X1 + 2*X2
print(y.shape)
# y = y.reshape(30,1)

# write your code here
xnew = [X1.reshape(900,1),X2.reshape(900,1)]
xnew = np.array(xnew).reshape(900,2)
ynew = y.reshape(900,1)
# xnew = xnew.T
print(xnew.shape)
w = np.array([5.,-6.,-4.])
eps = 0.00001
lr = 0.1
indexs = [1,2]
# w,w_gds,error1,epoch =  gradient_descent(w,eps,xnew,ynew,lr,indexs)
num=900
w = np.array([5.,-6.,-4.])
eps = 0.00001
lr = 0.1
X1 = X1.reshape(900,1)
X2 = X2.reshape(900,1)
y = w[0] + 1*X1 +2*X2
error1 = 1000001.
error2 = 1000000.
error_gd=[]
w1=[]
w2=[]
epoch=0
while abs(error1-error2)>eps:
    epoch+=1
    # print(error1)
    y_pred = w[0] + w[1]*X1 + w[2]*X2
    w1.append(w[1])
    w2.append(w[2])
    error1 = np.sum((y-y_pred)**2)/num
    error_gd.append(error1)
    del_error_1 = -(np.sum(np.dot((y-y_pred).T,X1)))/num
    del_error_2 = -(np.sum(np.dot((y-y_pred).T,X2)))/num
    w[1] = w[1] - lr * del_error_1
    w[2] = w[2] - lr * del_error_2
    # print(w)
    y_pred = w[0] + w[1]*X1 + w[2]*X2
    error2 = np.sum((y-y_pred)**2)/num

print(w)
print(epoch)
plt.contour(w3,w4,J(aa0,aa1,x1,y1),60)
# plt.show()
plt.plot(w1, w2, 'black')
plt.plot(w[1],w[2], 'orange', marker = 'X')
plt.show()