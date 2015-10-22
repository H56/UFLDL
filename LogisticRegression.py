import numpy as np
import matplotlib.pyplot as plt
import MNIST


class LogisticRession:
    theta = None
    alpha = 0.0001

    def __init__(self, theta=None, alpha=0.0001):
        self.theta = theta
        self.alpha = alpha

    def delta(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, images):
        if self.theta is None:
            self.theta = np.zeros((images.shape[0], 1))
        return self.delta(np.dot(self.theta.T, images))

    def cost_function(self, X, y):
        return -(np.dot(y, self.h(X).T))

    def gradient(self, X, y):
        return np.dot(X, (self.h(X) - y).T)

    def train(self, images, labels, times=100):
        cost = []
        for i in range(0, times):
            gradient = self.gradient(images, labels)
            self.theta -= self.alpha * gradient
            cost.append(self.cost_function(images, labels))
        return cost
train_image_filename = input('train image filenaem:')
train_label_filename = input('train label filename:')
reader = MNIST.MNIST(train_image_filename, train_label_filename)
train_image_mat, train_label_mat = reader.read_filter((0, 1))
logisticRegression = LogisticRession()
train_image_mat = train_image_mat.reshape((-1, train_image_mat[0].shape[0] * train_image_mat[0].shape[1])).T
cost = logisticRegression.train(np.vstack((train_image_mat, np.ones((1, train_image_mat.shape[1])))), train_label_mat)
test_image_filename = input('test image filename:')
test_image_filenaem = input('test image filename:')
reader = MNIST.MNIST(train_image_filename, train_label_filename)
test_image_mat, test_label_mat = reader.read_filter((0, 1))
test_image_mat = test_image_mat.reshape((-1, test_image_mat[0].shape[0] * test_image_mat[0].shape[1])).T
labels = logisticRegression.h(np.vstack((test_image_mat, np.ones((1, test_image_mat.shape[1])))))
plt.figure(1)
plt.plot(cost)
print((labels == test_label_mat).sum() / labels.size)
print((labels == [1 if i > 0.5 else 0 for i in (set for set in test_label_mat)]).sum() / labels.size)
plt.show()
