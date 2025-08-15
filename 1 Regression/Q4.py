import numpy as np
import pandas as pd
dt = pd.read_csv("salary_pred_data1.csv")
data = dt.to_numpy()
test = 100
train = 900
data_train = data[0:900,:]
y_train = data_train[:,[5]]
x_train = data_train[:,[0,1,2,3,4]]
x_train = x_train.T
#Inverse Matrix Method
w_pred_1=reg.mat_inv(y_train,x_train)
y_pred = x_train.T @ w_mat
#Gradient Decent Method
lr = 0.01
w_pred_2,err = reg.Regression_grad_des(x_train,y_train,lr)
data_test = data[900:1000,:]
y_test = data_test[:,[5]]
x_test = data_test[:,[0,1,2,3,4]]
x_test = x_test.T
error1=reg.error(w_pred_1,y_test,x_test)/((np.max(y_test)-np.mean(y_test))**2)
error2=reg.error(w_pred_2,y_test,x_test)/((np.max(y_test)-np.mean(y_test))**2)
print(w_pred_1)
print('Normalized testdata error using Inverse matrix metehod : ',error1,'\n')