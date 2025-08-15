import numpy as np
import matplotlib.pyplot as plt

N=1000
x = np.random.rand(N)
y=x+10
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming t%matplotlib inlinehe y axis 
plt.ylabel('y - axis') 

  
# function to show the plot 
plt.show()