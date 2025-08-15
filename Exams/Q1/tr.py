import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from Regression import Regression

def errorUsingW(w, x, y):
    predictions = x @ w
    return (np.sum(np.square(y - predictions))) / (2 * y.shape[0])
def addOnesColumn(x):
    const_ar = np.array([[1] for i in range(x.shape[0])])
    x = np.append(const_ar, x, axis=1)
    return x
def performMatrixInversion(x, y):
    x = addOnesColumn(x)
    w = ((np.linalg.inv(x.T @ x)) @ x.T @ y)
    newY = x @ w
    return w, newY, errorUsingW(w, x, y)
def multivar_reg(x, y, KFOLD):
    for fold_No in range(KFOLD):
        print("K_no :: ", fold_No)
        blockSize = int(((x.shape[0] + KFOLD - 1) / KFOLD))
        # print("size : ", blockSize)

        blockStartPoint = fold_No * blockSize
        blockEndPoint = blockStartPoint + blockSize

        trainingX = np.concatenate((x[:blockStartPoint, :], x[blockEndPoint:, :]))
        trainingY = np.concatenate((y[:blockStartPoint, :], y[blockEndPoint:, :]))

        testX = x[blockStartPoint:blockEndPoint, :]
        testY = y[blockStartPoint:blockEndPoint, :]

        #transform testX (add colums of ones)
        testX = addOnesColumn(testX)


        print("Using Matrix Inverse Method :")
        W, newY, leastError = performMatrixInversion(trainingX, trainingY)
        testError = errorUsingW(W, testX, testY)
        print("W : ", W.T, "  error : ", testError)
        print("----------------------------------------------------")


with open('quiz1_dataset.csv') as file:
    data = list(csv.reader(file, delimiter=','))
data = np.array(data)
data = data.astype(np.float)

KFOLD = 5

x  = data[:, 0:3]
y  = data[:, 3:4]

multivar_reg(x,y,5)









