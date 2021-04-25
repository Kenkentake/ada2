from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(XMIN, XMAX, SAMPLE_SIZE):
    x = np.linspace(start=XMIN, stop=XMAX, num=SAMPLE_SIZE)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=SAMPLE_SIZE)
    return x, target + noise


def calc_design_matrix(x):
    SAMPLE_SIZE = len(x)
    phi = np.empty(shape=(SAMPLE_SIZE, 31))  # design matrix
    phi[:, 0] = 1.
    phi[:, 1::2] = np.sin(x[:, None] * np.arange(1, 16)[None] / 2)
    phi[:, 2::2] = np.cos(x[:, None] * np.arange(1, 16)[None] / 2)
    return phi


# create sample
SAMPLE_SIZE = 50
XMIN, XMAX = -3, 3
x, y = generate_sample(XMIN=XMIN, XMAX=XMAX, SAMPLE_SIZE=SAMPLE_SIZE)

# calculate design matrix
phi = calc_design_matrix(x)

# solve the least square problem
theta = np.linalg.solve(np.dot(phi.T, phi), np.dot(phi.T, y[:, None]))

# create data to visualize the prediction
X = np.linspace(start=XMIN, stop=XMAX, num=5000)
Phi = calc_design_matrix(X)
prediction = np.dot(Phi, theta)

# visualization
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)
plt.show()
plt.savefig('result_linear_model.png')
