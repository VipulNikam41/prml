import numpy as np
import csv
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------
dataset = np.genfromtxt( "quiz1_dataset.csv" , delimiter=',')
x = dataset[:, :-1]
y = dataset[:, -1]
y=y.reshape(487,1)
print(x.shape,y.shape)
# ---------------------------------------------------------------------------------------------
#feature scaling using min-max normalization
ma=np.max(x,axis=0)
mi=np.min(x,axis=0)
for i in range(3):
    x[:,i]=(x[:,i]-mi[i])/(ma[i]-mi[i])
# -------------------------------------------------------------------------------------------------------
plt.figure(1)
plt.scatter(x[:,0], y)
plt.figure(2)
plt.scatter(x[:,1], y)
plt.figure(3)
plt.scatter(x[:,2], y)
# -------------------------------------------------------------------------------------------------------------
class regr:
  # Constructor
  def __init__(self, name='reg'):
    self.name = name  # Create an instance variable

  def grad_update(self,w_old,lr,y,x,m):
   # write your code here
    w_pred=w_old + (2*lr/m)* np.dot(x, y - np.dot(x.T,w_old))
    return w_pred

  def error(self,w,y,x):
    return  (1/x.shape[1])*np.sum(np.square(y - np.dot(x.T,w)))
 
  def mat_inv(y,x_aug):
    return np.dot( np.linalg.inv( np.dot(x_aug,x_aug.T)) , np.dot( x_aug, y) )

    # by Gradien descent
  def Regression_grad_des(self,x,y,lr):
    # write your code here
    m=x.shape[1]
    w=np.zeros((x.shape[0],1))
    err=[]
    for i in range(m):
        w=self.grad_update(w,lr,y,x,m)
        #err.append(self.error(w,y,x))
    return w,err
# -----------------------------------------------------------------------------------------------------------------
x0 = x[:,0]
x1 = x[:,1]
x2 = x[:,2]
x0=x0.reshape(1,487)
x1=x1.reshape(1,487)
x2=x2.reshape(1,487)
print(x0.shape,x1.shape)

x0=np.concatenate((np.ones((1,x0.shape[1])), x0),axis=0)
x1=np.concatenate((np.ones((1,x1.shape[1])), x1),axis=0)
x2=np.concatenate((np.ones((1,x2.shape[1])), x2),axis=0)

w_opt=regr.mat_inv(y,x0)
print("w1 using la ",w_opt)
# by Gradien descent
lr=0.02
w_pred=[[0], [0]]
for i in range(10000):
    w_pred +=  (2*lr/487)* (x0@ (y - x0.T @ w_pred))
print("w1 using grad. desc. ", w_pred)
y0=x0.T @ w_opt

plt.figure(1)
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],y0,color='orange')

w_opt=regr.mat_inv(y,x1)
print(w_opt)
# by Gradien descent
lr=0.001
w_pred=[[0], [0]]
for i in range(10000):
    w_pred +=  (2*lr/487)* (x1@ (y - x1.T @ w_pred))
print(w_pred)
y1=x1.T @ w_opt

plt.figure(2)
plt.scatter(x[:,1],y)
plt.scatter(x[:,1],y1,color='orange')

w_opt=regr.mat_inv(y,x2)
print(w_opt)
# by Gradien descent
lr=0.002
w_pred=[[0], [0]]
for i in range(100000):
    w_pred +=  (2*lr/487)* (x2@ (y - x2.T @ w_pred))
print(w_pred)
y2=x2.T @ w_opt

plt.figure(3)
plt.scatter(x[:,2],y)
plt.scatter(x[:,2],y2,color='orange')

# ------------------------------------------------------------------------------------------------------------------
def data_transform(X,degree): 
 # write your code here
    if degree==0:
        X_new=np.ones((1,X.shape[1]),float)
        return X_new
    if degree==1:
        return X
    if degree>1:
        for i in range(degree-1):
            X=np.append(X,[X[1]],axis=0)
            X[i+2]=np.multiply(X[i+2],X[i+1])
    return X
# -----------------------------------------------------------------------------------------------------------
x0 = x[:,0]
x1 = x[:,1]
x2 = x[:,2]
x0=x0.reshape(1,487)
x1=x1.reshape(1,487)
x2=x2.reshape(1,487)
print(x0.shape,x1.shape)

x0=np.concatenate((np.ones((1,x0.shape[1])), x0),axis=0)
x1=np.concatenate((np.ones((1,x1.shape[1])), x1),axis=0)
x2=np.concatenate((np.ones((1,x2.shape[1])), x2),axis=0)

x0 = data_transform(x0,3)
w_opt=regr.mat_inv(y,x0)
print("w1 using la ",w_opt)
y0=x0.T @ w_opt

plt.figure(1)
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],y0,color='orange')

x1 = data_transform(x1,3)
w_opt=regr.mat_inv(y,x1)
print("w1 using la ",w_opt)
y1=x1.T @ w_opt

plt.figure(2)
plt.scatter(x[:,1],y)
plt.scatter(x[:,1],y1,color='orange')

x0 = data_transform(x2,1)
w_opt=regr.mat_inv(y,x2)
print("w1 using la ",w_opt)
y2=x2.T @ w_opt

plt.figure(3)
plt.scatter(x[:,2],y)
plt.scatter(x[:,2],y2,color='orange')
# ------------------------------------------------------------------------------------------------
from mpl_toolkits import mplot3d

#X, Y = np.meshgrid(x[:,0].flatten(), x[:,1].flatten())
plt.figure(1)
ax = plt.axes(projection='3d')
#ax.plot_surface(X,Y,y)
ax.scatter(x[:,0], x[:,1], y)
ax.set_xlabel('x0')
ax.set_ylabel('x1')

#X, Y = np.meshgrid(x[:,1], x[:,2])
plt.figure(2)
ax = plt.axes(projection='3d')
ax.scatter(x[:,1], x[:,2], y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')

#X, Y = np.meshgrid(x[:,0], x[:,2])
plt.figure(3)
ax = plt.axes(projection='3d')
ax.scatter(x[:,0], x[:,2], y)
ax.set_xlabel('x0')
ax.set_ylabel('x2')
# --------------------------------------------------------------------------------------------------
x01 = x[:,0:2]
x12 = x[:,1:3]
x02 = x[:,[0,2]]
x01 = x01.T
x12 = x12.T
x02 = x02.T
x01=np.concatenate((np.ones((1,x01.shape[1])), x01),axis=0)
x12=np.concatenate((np.ones((1,x12.shape[1])), x12),axis=0)
x02=np.concatenate((np.ones((1,x02.shape[1])), x02),axis=0)

reg=regression()
w_opt=reg.mat_inv(y,x01)
print(w_opt)

# by Gradien descent
lr=0.003
#w_pred,err=reg.Regression_grad_des(x01,y,lr)
w_pred=[[0], [0], [0]]
for i in range(100000):
    w_pred +=  (2*lr/487)* (x01@ (y - x01.T @ w_pred))
print(w_pred)
print()

reg=regression()
w_opt=reg.mat_inv(y,x12)
print(w_opt)

# by Gradien descent
lr=0.0015
#w_pred,err=reg.Regression_grad_des(x01,y,lr)
w_pred=[[0], [0], [0]]
for i in range(1000000):
    w_pred +=  (2*lr/487)* (x12@ (y - x12.T @ w_pred))
print(w_pred)
print()

reg=regression()
w_opt=reg.mat_inv(y,x02)
print(w_opt)

# by Gradien descent
lr=0.0025
#w_pred,err=reg.Regression_grad_des(x01,y,lr)
w_pred=[[0], [0], [0]]
for i in range(1000000):
    w_pred +=  (2*lr/487)* (x02@ (y - x02.T @ w_pred))
print(w_pred)
# ----------------------------------------------------------------------------------------------------------
X=x.T
X=np.concatenate((np.ones((1,X.shape[1])), X),axis=0)
w_opt=reg.mat_inv(y,X)
print(w_opt)