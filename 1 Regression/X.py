import numpy as np
import csv
# write your code here
x = np.genfromtxt("salary_pred_data_x.csv" , delimiter=',', skip_header=1)
x=x.T
print( x.shape)
aug_x=np.concatenate((np.ones((1,x.shape[1])), x),axis=0)
print (aug_x.shape)
y = np.genfromtxt("salary_pred_data_y.csv", delimiter=',')
y = y.reshape((3012,1))
print(np.any(np.isnan(y)))
mini=np.amin(y)
#y = y/mini
print(mini, y.shape)

aug_x_train = aug_x[:, : 900]
print(aug_x_train.shape)
aug_x_test = aug_x[:, 900:]
y_train = y[:900]
print(y_train.shape)
y_test = y[900:]

w_pred=[[0], [0], [0], [0], [0], [0]]

lr=0.0001

for i in range(100000):
    w_pred +=  (2*lr/1000)* (aug_x_train@ (y_train - aug_x_train.T @ w_pred))
#print(w_pred)
reg=regression()
#w_pred=reg.mat_inv(y_train,aug_x_train)
print(w_pred)
y_pred = aug_x_test.T @ w_pred

# mean square error (testing) (normalized) #############

error=reg.error(w_pred,y_test,aug_x_test)/((np.max(y_test)-np.mean(y_test))**2)
print('Normalized testing error=',error,'\n')

error=reg.error(w_pred,y_train,aug_x_train)/((np.max(y_train)-np.mean(y_train))**2)
print('Normalized training error=',error,'\n')

print('predicted salary=',y_pred[0:3],'\n')
print('actual salary=',y_test[0:3])