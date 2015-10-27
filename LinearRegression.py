import numpy as np
import matplotlib.pyplot as plt
from sys import *


class LinearRegression:
    # """A Simple Linear Regression Sample"""

    theta = None
    alpha = 0

    def __init__(self, alpha=0.0001, theta=None):
        self.theta = theta
        self.alpha = alpha

    def cost_function(self, X, y):
        if self.theta is None:
            self.theta = np.zeros(X.shape[0]).T
        tmp = np.dot(self.theta.T, X) - y
        return (np.dot(tmp, tmp.T)).sum() / 2.

    def h(self, X):
        return np.dot(self.theta.T, X)

    def gradient_descent(self, X, y):
        if self.theta is None:
            self.theta = np.zeros(X.shape[0]).T
        return np.dot(X, (self.h(X) - y).T)

    def train(self, X, y):
        old_cost = 0
        if self.theta is None:
            self.theta = np.zeros(X.shape[0]).T
        # while abs(old_cost - self.cost_function(X, y)) > delta:
        cost = [self.cost_function(X, y)]
        for i in range(0, 10000):
            self.theta -= self.alpha * self.gradient_descent(X, y)
            cost.append(self.cost_function(X, y))
        return self.theta, cost

def normalMat(X):
    for i in range(0, X.shape[0]):
        max = abs(X[i]).max()
        if max != 0:
            X[i] /= max

path = input('File Path:')
fd = open(path, 'r')
source = []
# D:\OneDrive\housing_data.txt

price = []
for line in fd:
    tmp = line[: -1].split()
    source.append(np.append(1., [np.double(i) for i in tmp[0: len(tmp) - 1]]))
    price.append(np.double(tmp[-1]))
fd.close()

testSet = np.array(source[0: 100]).T
normalMat(testSet)
testy = np.array(price[0: 100]).T
trainSet = np.array(source[100: -1]).T
normalMat(trainSet)
trainy = np.array(price[100: -1])

LR = LinearRegression()
(theta, cost) = LR.train(trainSet, trainy)
y = LR.h(testSet)
plt.figure(1)
plt.plot(np.arange(len(cost)), cost)

x = np.arange(0, 500, 5)
plt.figure(2)
plt.plot(x, testy, 'r+')
plt.plot(x, y, 'b*')
plt.margins(0.2)
plt.show()
