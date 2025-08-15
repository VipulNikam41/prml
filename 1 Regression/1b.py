import numpy as np
import matplotlib.pyplot as plt

N=100
x = np.random.rand(N)
z = np.random.rand(N)

ycor=x+10+0.1*z
plt.scatter(x,ycor) 
  

plt.show()