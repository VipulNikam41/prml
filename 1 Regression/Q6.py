from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df = pd.read_csv("./drive/My Drive/Copy of salary_pred_data1.csv")
print(df.head)

df.columns


x = df[['Level of city', 'Years of experiance', 'Age', 'Level of education',
       'Job profile']]
y = df[['Salary']]
# print(y)
x = np.concatenate((np.ones((x.shape[0],1)), x),axis=1)
# print(np.ones((1,x.shape[0])).shape)
x_train,y_train,x_test,y_test = x[:900],y[:900],x[900:1000],y[900:1000]
# print(x_test.T)
reg = regression()
# w_pred=reg.mat_inv(y_train,x_train.T)
# print(w_pred)
# print(x_test.T.shape,w_pred.shape)
x_test = np.array(x_test)
# x_test_t = np.concatenate((np.ones((1,x_test.T.shape[1])), x_test.T),axis=0)
w_pred=reg.mat_inv(y_train,x_train.T)
print(x_test[0:3]@w_pred)
y_pred = x_test@w_pred
print(y_pred)
error=reg.error(w_pred,y_test,x_test.T)/((np.max(y_test)-np.mean(y_test))**2)

print('Normalized testing error=',error,'\n')

print('predicted salary=',y_pred[0:3],'\n')
print('actual salary=',y_test[0:3])


error=reg.error(w_pred,y_train,x_train.T)/((np.max(y_train)-np.mean(y_train))**2)

print('Normalized training error=',error,'\n')

# print(x_train)
x_train = np.array(x_train,dtype=np.float64)
y_train = np.array(y_train,dtype=np.float64)
x_test = np.array(x_test,dtype=np.float64)
y_test = np.array(y_test,dtype=np.float64)
# print(x_train)
w_pred_gd,err1 = reg.Regression_grad_des(x_train.T,y_train,0.0001)
print(w_pred_gd)
y_pred = x_test@w_pred_gd
# print(y_pred)
error=reg.error(w_pred_gd,y_test,x_test.T)/((np.max(y_test)-np.mean(y_test))**2)

print('Normalized testing error=',error,'\n')

print('predicted salary=',y_pred[0:3],'\n')
print('actual salary=',y_test[0:3])



error=reg.error(w_pred_gd,y_train,x_train.T)/((np.max(y_train)-np.mean(y_train))**2)

print('Normalized training error=',error,'\n')


import numpy as np

# write your code here

# mean square error (testing) (normalized) #############

error=reg.error(w_pred,y_test,aug(x_test))/((np.max(y_test)-np.mean(y_test))**2)

print('Normalized testing error=',error,'\n')

print('predicted salary=',y_pred[0:3],'\n')
print('actual salary=',y_test[0:3])
