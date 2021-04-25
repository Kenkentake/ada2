from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


class LeastSquareKernel:
    def __init__(self, x_train, y_train, x_val, y_val, h, l):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.h = h
        self.l = l

    def calc_design_matrix(self, x, c):
        return np.exp(-(x[None] - c[:, None]) ** 2 /(2 * self.h ** 2))

    def calc_theta(self, k, y):
        return np.linalg.solve(k.T.dot(k) + self.l * np.identity(len(k)), k.T.dot(y[:, None]))

    def calc_loss(self, theta, k, y):
        return np.mean((k.dot(theta) - y) ** 2)

    def get_loss(self):
        k_train = self.calc_design_matrix(x_train, x_train)
        theta = self.calc_theta(k_train, y_train)
        k_val = self.calc_design_matrix(x_train, x_val)
        return self.calc_loss(theta, k_val, y_val)

def generate_sample(XMIN, XMAX, SAMPLE_SIZE):
    x = np.linspace(start=XMIN, stop=XMAX, num=SAMPLE_SIZE)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=SAMPLE_SIZE)
    return x, target + noise

# h and l's candidates
h_list = np.arange(0.01, 1.0, 0.01).tolist()
l_list = np.arange(0.01, 1.0, 0.01).tolist()
loss_list = []
param_list = []
# create sample
SAMPLE_SIZE = 50
XMIN, XMAX = -3, 3
x, y = generate_sample(XMIN=XMIN, XMAX=XMAX, SAMPLE_SIZE=SAMPLE_SIZE)

for h in h_list:
    for l in l_list:
        tmp_losses = []
        for i_cv in range(5):
            x_list = np.split(x, 5)
            y_list = np.split(y, 5)
            x_val = x_list.pop(i_cv)
            y_val = y_list.pop(i_cv)
            x_train = np.concatenate(x_list)
            y_train = np.concatenate(y_list)
            tmp_loss = LeastSquareKernel(x_train, y_train, x_val, y_val, h, l).get_loss()
            tmp_losses.append(tmp_loss)
        mean_loss = sum(tmp_losses) / len(tmp_losses)
        loss_list.append(mean_loss)
        param_list.append([h, l])

# get best params and loss
loss_array = np.array(loss_list)
best_index = np.argmin(loss_array)
best_params = param_list[best_index]
best_loss = loss_list[best_index]
print(f'best_params are h = {best_params[0]}, lambda = {best_params[1]}')
print(best_loss)

# best_params are h = 0.78, lambda = 0.01
# loss is 0.057651799362229125

        
 
# # visualization
# plt.clf()
# plt.scatter(x, y, c='green', marker='o')
# plt.plot(X, prediction, label='pred')
# plt.plot(X, np.sin(np.pi*X) / (np.pi*X) + 0.1 * X, label='target')
# plt.legend()
# plt.savefig('result_kernel_model.png')

