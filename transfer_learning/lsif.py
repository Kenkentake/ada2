import numpy as np
from scipy.stats import norm
import matplotlib

import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n=100, n2=100):
    train = np.random.normal(loc=0., scale=.50, size=n2)
    test = np.random.normal(loc=1., scale=.25, size=n)
    return train, test


def lsif(train, test):
    n, n2 = len(train), len(test)
    xx = (test[None] - test[:, None]) ** 2
    ux = (train[None] - test[:, None]) ** 2
    sigma = .1
    k = np.exp(-xx / sigma)
    r = np.exp(-ux / sigma)
    alpha = np.linalg.solve(r.dot(r.T) / n + .1 * np.eye(n2),
                            np.mean(k, axis=1))
    return alpha.dot(r)


def visualize(train, w):
    plt.clf()
    plt.xlim(-2., 2.)
    plt.ylim(0, 20)
    x = np.linspace(-2., 2., 100)
    plt.plot(x, norm.pdf(x, 0., .50),
             label='$p_{\\mathrm{train}}(x)$', color='blue')
    plt.plot(x, norm.pdf(x, 1., .25),
             label='$p_{\\mathrm{test}}(x)$', color='black')
    plt.scatter(train, w, marker='x', color='red')
    plt.plot(x, norm.pdf(x, 1., .25) / norm.pdf(x, 0., .50),
             label='$w(x)$', color='red')
    plt.legend()
    plt.savefig('lecture8-p36.png')


n, n2 = 100, 200
train, test = generate_data(n, n2)
w = lsif(train, test)
visualize(train, w)
