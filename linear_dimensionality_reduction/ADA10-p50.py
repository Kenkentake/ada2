import numpy as np
import matplotlib
from scipy.linalg import eig

import matplotlib.pyplot as plt

np.random.seed(46)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y):
    m1 = np.mean(x[y == 1], axis=0, keepdims=True)
    m2 = np.mean(x[y == 2], axis=0, keepdims=True)
    x1 = x[y == 1] - m1
    x2 = x[y == 2] - m2
    n1 = np.sum(np.where(y == 1, 1., 0.))
    n2 = np.sum(np.where(y == 2, 1., 0.))
    V, T = eig(n1 * m1.T.dot(m1) + n2 * m2.T.dot(m2), x1.T.dot(x1) + x2.T.dot(x2))
    return T[np.argsort(V)[:-2:-1]]



def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 9, np.array([-T[:, 1], T[:, 1]]) * 9, 'k-')
    plt.legend()
    plt.savefig('lecture11-h1.png')


sample_size = 100
# x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
T = fda(x, y)
visualize(x, y, T)
