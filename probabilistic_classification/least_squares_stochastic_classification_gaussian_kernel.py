import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_data(sample_size, n_class):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y

def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

def visualize(x, y, theta, h):
    X = np.linspace(-5., 5., num=100)
    K = calc_design_matrix(x, X, h)

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)
    logit = K.dot(theta)
    unnormalized_prob = np.exp(logit - np.max(logit, axis=1, keepdims=True))
    prob = unnormalized_prob / unnormalized_prob.sum(1, keepdims=True)

    plt.plot(X, prob[:, 0], c='blue')
    plt.plot(X, prob[:, 1], c='red')
    plt.plot(X, prob[:, 2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')
    plt.savefig('lecture7-p17.png')

l = 0.3
sample_size = 90
n_class = 3
phi = np.zeros((sample_size, n_class))
x, y = generate_data(sample_size, n_class)
for i in range(n_class):
    phi[:, i] = (y ==i)
k = calc_design_matrix(x, y, h=2.)
theta = np.linalg.solve(
        k.T.dot(k) + l*np.identity(len(k)), k.T.dot(phi))
visualize(x, y, theta, h=2.)
