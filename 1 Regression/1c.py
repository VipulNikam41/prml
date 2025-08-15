import numpy as np
import random
import matplotlib.pyplot as plt
x=[]
N=1000
x = np.random.rand(N)
z = np.random.rand(N)

y_cor=x+10+0.1*z
w = []

for i in range(1000):
   
    k = random.uniform(-5, 7)
    w.append(k)
w.sort()   
error=[]
for i in range(1000):
    y_pred=[]
    error.append(0)
    for j in range(1000):
        y_pred.append(w[i]*x[j]+10)
        error[i]+=(y_cor[j]-y_pred[j])**2
    error[i]/=1000
        
plt.figure(1)          
plt.plot(w, error)

min_in=error.index(min(error))

y_best=[]
for i in range(1000):
    y_best.append(w[min_in]*x[i] + 10)
plt.figure(2)
plt.scatter(x,y_cor)
plt.plot(x,y_best, color='orange')