import MNIST
from scipy import sparse

__author__ = 'HuPeng'
import numpy as np
import matplotlib.pyplot as plt

class SoftmaxRegression:
    theta = None
    alpha = None

    def __init__(self, dimension=10, alpha=5.85e-6, theta=None):
        self.theta = theta
        self.alpha = alpha
        self.dimension = dimension

    def h(self, x):
        if self.theta is None:
            self.theta = np.random.random_sample((x.shape[0], self.dimension)) * self.alpha
        result = np.exp(np.dot(self.theta.T, x))
        return np.array([result[:, i] / result[:, i].sum() for i in range(0, result.shape[1])]).T

    def cost(self, x, y):
        if self.theta is None:
            self.theta = np.random.random_sample((x.shape[0], self.dimension)) * self.alpha
        return -sum([np.log(np.exp(np.dot(self.theta[:, y[i]], x[:, i].T)) / np.exp(np.dot(self.theta.T, x[:, i].T)).sum()) for i in range(0, x.shape[1])])

    def gradient(self, x, y):
        w = np.exp(np.dot(self.theta.T, x))
        w /= w.sum(axis=0)
        ones = sparse.coo_matrix((np.ones(x.shape[1]), (y, np.arange(x.shape[1]))), shape=w.shape).toarray()
        ct = -np.dot(ones.reshape(-1), np.log(w.reshape(-1)))
        gdt = np.dot(x, (w - ones).T)
        return gdt, ct

    def train(self, x, y, times=1000):
        if self.theta is None:
            self.theta = np.random.random_sample((x.shape[0], self.dimension)) * self.alpha
        cost = []
        for i in range(0, times):
            gradient, ct = self.gradient(x, y)
            self.theta -= self.alpha * gradient
            cost.append(ct)
            print(i)
        return cost

    def compute(self, test):
        h = self.h(test)
        return [h[:, i].argmax() for i in range(0, h.shape[1])]

train_image_filename = 'dataSet/MNIST/train-images.idx3-ubyte'# input('train image filenaem:')
train_label_filename = 'dataSet/MNIST/train-labels.idx1-ubyte' # input('train label filename:')
reader = MNIST.MNIST(train_image_filename, train_label_filename)
train_image_mat, train_label_mat = reader.read_standardize()
softmaxRegression = SoftmaxRegression()
cost = softmaxRegression.train(train_image_mat, train_label_mat, 200)

test_image_filename = 'dataSet/MNIST/t10k-images.idx3-ubyte' # input('train image filenaem:')
test_label_filename = 'dataSet/MNIST/t10k-labels.idx1-ubyte' # input('train label filename:')
reader = MNIST.MNIST(test_image_filename, test_label_filename)
test_image_mat, test_label_mat = reader.read_standardize()
test = softmaxRegression.compute(test_image_mat)
print(str((test == test_label_mat).sum() / len(test) * 100) + '%')
plt.figure(1)
plt.plot(range(0, len(cost)), cost)
plt.show()
