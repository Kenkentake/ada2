import numpy as np
import matplotlib

import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + .5, -np.sin(x)], axis=1) * 10.
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - .5, np.sin(x)], axis=1) * 10.
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    y[0] = 1
    y[-1] = -1
    return x, y


def lrls(x, y, h=1., l=1., nu=1.):
    """

    :param x: data points
    :param y: labels of data points
    :param h: width parameter of the Gaussian kernel
    :param l: weight decay
    :param nu: Laplace regularization
    :return:
    """
    n = len(x)
    phi = np.exp(-np.sum((x[None] - x[:, None]) ** 2, axis=2) / (2 * h ** 2))
    # only labeled data
    phi_tilde = phi[y != 0]
    W = np.exp(-np.sum((x[None] - x[:, None]) ** 2, axis=2) / (2 * h ** 2))
    L = np.diag(np.sum(W, axis=1)) - W
    return np.linalg.solve(
            phi_tilde.T.dot(phi_tilde) + l * np.eye(n) + 2 * nu * phi.T.dot(L).dot(phi),
            phi_tilde.T.dot(y[y != 0])
    )

def visualize(x, y, theta, h=1.):
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(-20., 20.)
    plt.ylim(-20., 20.)
    grid_size = 100
    grid = np.linspace(-20., 20., grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    k = np.exp(-np.sum((x.astype(np.float32)[:, None] - mesh_grid.astype(
        np.float32)[None]) ** 2, axis=2).astype(np.float64) / (2 * h ** 2))
    plt.contourf(X, Y, np.reshape(np.sign(k.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='$.$', c='black')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$X$', c='red')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='$O$', c='blue')
    plt.savefig('lecture8-h1.png')


x, y = generate_data(n=200)
lrls(x, y, h=1.)
theta = lrls(x, y, h=1.)
visualize(x, y, theta)
