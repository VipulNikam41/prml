# write your code here
import numpy as np
import matplotlib.pyplot as plt

w1 = [0 for _ in range(1000)]
w1[0] = -4
w0 = 0
learn_rate = 0.01
error1 = [0 for _ in range(1000)]
error_gradient = 0
y_pred1 = [0 for _ in range(1000)]

for j in range(1000-1):
  for i in range(1000):
    y_pred1[i] = w0 + w1[j]*x_values[i]

  for i in range(1000):
    error1[j] += (y_corrupt[i] - y_pred1[i])*(y_corrupt[i] - y_pred1[i])
  error1[j] /= 1000

  for i in range(1000):
    error_gradient += (-2)*((y_corrupt[i] - y_pred1[i])*x_values[i])
  error_gradient /= 1000

  w1[j+1] = w1[j] - learn_rate*error_gradient

fig, axs1 = plt.subplots(1)
fig, axs2 = plt.subplots(1)
axs1.plot(search_w1, error)
axs1.plot(w1, error1, "black")

axs1.plot(w1[998], error1[998],"r*")
axs2.scatter(x_values, y_corrupt, s=5)
axs2.plot(x_values, y_pred1, "red")
plt.show()