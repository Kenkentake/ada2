from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

class AdmmKernelNorm1():
    def __init__(self, h, l, x, y):
        self.h = h
        self.l = l
        self.x = x
        self.y = y
        
    def calc_design_matrix(self, c):
        return np.exp(-(self.x[None] - c[:, None]) ** 2 / (2 * self.h ** 2))
        
    def calc_loss(self, theta, k):
        return np.mean((k.dot(theta) - self.y) ** 2) / 2 + l * np.linalg.norm(theta, ord=1)

    def init_params(self):
        return np.random.rand(), np.random.rand(), np.random.rand()

    def admm_step(self, max_iter):
        k = self.calc_design_matrix(self.x)
        theta_next, z_next, u_next = self.init_params()
        for i in range(max_iter):
            theta = theta_next
            z = z_next
            u = u_next
            theta_next = np.linalg.solve(k.T.dot(k) + np.identity(len(k)), 
                                            k.T.dot(y[:, None]) + z - u)
            z_next = np.maximum(np.zeros(len(k)), theta_next + u - self.l * np.ones(len(k)))
            u_next = u + theta_next - z_next
            loss = self.calc_loss(theta_next, k)

sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)
h = 0.78
l = 0.01
loss = AdmmKernelNorm1(h, l, x, y).admm_step(max_iter=200)
