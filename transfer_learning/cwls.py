import numpy as np
import matplotlib

import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 0
    y[n_positive:] = 1
    return x, y


def cwls(train_x, train_y, test_x):
    pass

def cwls(train_x, train_y, test_x):
    train_x = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    test_x = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    xx = np.sqrt(np.sum((train_x[:, None] - train_x[None]) ** 2, axis=2))
    xX = np.sqrt(np.sum((train_x[:, None] - test_x[None]) ** 2, axis=2))
    ratio_of_class = np.empty(2)
    b = np.empty(2)
    A = np.empty((2, 2))
    for i in [0, 1]:
        ratio_of_class[i] = np.mean(np.where(train_y == i, 1., 0.))
        b[i] = np.mean(xX[train_y == i])
        for j in [0, 1]:
            A[i, j] = np.mean(
                xx[(train_y == i)[:, None] * (train_y == j)[None]])
    pi = (A[0, 1] - A[1, 1] - b[0] + b[1]) / (2 * A[0, 1] - A[0, 0] - A[1, 1])
    pi = np.clip(pi, 0., 1.)
    pi = np.array([pi, 1 - pi])
    w = pi[train_y] / ratio_of_class[train_y]
    z = np.where(train_y == 0, 1., -1.)
    return np.linalg.solve(train_x.T.dot(w[:, None] * train_x),train_x.T.dot(w * z))

def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1],
                    marker='$O$', c='blue')
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
                    marker='$X$', c='red')
        plt.savefig('lecture8-h3-{}.png'.format(name))


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
