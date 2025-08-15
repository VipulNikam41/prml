# video reference: https://youtu.be/Qa_FI92_qo8
import numpy as np 

NUM_PTS = 5

y = np.random.randint(0, 100, size=(NUM_PTS, 1))
x = np.random.randint(0, 100, size=(NUM_PTS))

y = np.array([x for x in range(5)])
x = np.array([x*2 for x in range(5)])

print(f'x: {x} \ny: {y}')

ones = np.ones(NUM_PTS)

x = np.vstack((ones, x)).transpose()
A = np.zeros(shape=(2,1))

# print(f'x: {x}', f'y: {y}', f'A: {A}', sep='\n')

A = np.linalg.inv(x.T @ x) @ x.T @ y

# print(f'A is now {A}')
print(f'm={A[1]}, c={A[0]}')














