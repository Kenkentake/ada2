import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)],
                          axis=1)


def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(
        np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)


def pca(x, n_components=1):
    x = x - np.mean(x, axis=0)
    w, v = np.linalg.eig(x.T.dot(x))
    return w[:n_components], v[:n_components, :]


n = 100
n_components = 1
x = data_generation1(n)
# x = data_generation2(n)
w, v = pca(x, n_components)

plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot(np.array([-v[:, 0], v[:, 0]]) * 9, np.array([-v[:, 1], v[:, 1]]) * 9)
plt.savefig('lecture10-p24.png')
