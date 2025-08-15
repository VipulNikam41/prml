#returns slope at co-ordinate w1 (x coordinate) in (sample_w1 vs error) graph
import numpy as np
import random
import matplotlib.pyplot as plt

def gradient_descent(x,y,w,e):
  m_curr = b_curr = 0
  iterations = 1000
  n = len(x)
  learning_rate = 0.001

  for i in range(iterations):
    y_pred = w * x + 10
    
    md = -2*sum(x, (y_cor-y_pred))/n
    w = w - learning_rate * md
   # plt.plot(w, error)
    e.append(w)
  return e;

def slope(cur_w1):
    t = 0
    for i in range(1000):
        t += (2/1000) * (cur_w1*x[i]+w0 - currupted_y[i])*x[i]
    return t

#returns the value of cost function for the value of choosen w1
def costOf(cur_w1):
    cost = 0
    for i in range(1000):
        cost += pow(cur_w1*x[i]+w0 - currupted_y[i],2)/1000
    return cost
    


x=[]
N=1000
x = np.random.rand(N)
z = np.random.rand(N)

y_cor=x+10+0.1*z
w = []
e=[]
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
y=0
min_in=error.index(min(error))

y_best=[]
for i in range(1000):
    y_best.append(w[min_in]*x[i] + 10)
e = gradient_descent(x,y,w,e)
#plt.plot(e,error)
print(len(e))
print(len(error))
plt.figure(2)
plt.scatter(x,y_cor)
plt.plot(x,y_best, color='orange')
plt.show()



plt.plot(e,error)

startPoint = random.randint(-5,7)

alpha = 0.1


pathW = []
pathCost = []

pre_start = -100
while(abs(startPoint - pre_start) > 1e-4):
    pre_start = startPoint
    
    #values for graph
    pathW.append(pre_start)
    pathCost.append(costOf(pre_start))
    
    startPoint = startPoint - alpha * slope(startPoint)
    
plt.plot(pathW, pathCost)
plt.show()

final_w1 = startPoint
final_y = []
for i in range(1000):
    final_y.append(final_w1*x[i]+w0)
plt.plot(x,currupted_y,'.')
plt.plot(x,final_y,'.')