import glob
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

np.random.seed(1)

def load_csv(data_type):
    if data_type == 'train':
        csv_paths = glob.glob('digit/digit_train*.csv')
    if data_type == 'test':
        csv_paths = glob.glob('digit/digit_test*.csv')
    csv_paths = sorted(csv_paths)
    data_list = []
    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=',')
        data_list.append(data)
    return data_list


def generate_train_data(data_list, class_digit):
    x = np.concatenate(data_list, axis=0)
    y_list = []
    for i in range(10):
        if i == class_digit:
            y_list.append(np.ones(500))
            # y_list.append(np.ones(100))
        else:
            y_list.append(-np.ones(500))
            # y_list.append(-np.ones(100))
    y = np.concatenate(y_list)
    return x, y

def generate_test_data(data_list):
    x = np.array(data_list)
    x = np.transpose(x, (2, 1, 0))
    return x


def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))


def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data, 10.).T.dot(theta)


def decide_final_predict(test_data, theta_list, x_train):
    for i in range(10):
        binary_pred = predict(x_train, test_data, theta_list[i])
        if i == 0:
            binary_pred_matrix = binary_pred[None, :]
        else:
            binary_pred_matrix = np.insert(binary_pred_matrix, binary_pred_matrix.shape[0], binary_pred, axis=0)
    pred_class = np.argmax(binary_pred_matrix, axis=0)
    return pred_class


def build_confusion_matrix(train_data, x_test, theta_list):
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    for i in range(10):
        x_train, y_train = generate_train_data(train_data, i)
        transposed_x_test = np.transpose(x_test[:, :, i], (1, 0))
        pred_class = decide_final_predict(transposed_x_test, theta_list, x_train)
        for j in range(10):
            confusion_matrix[i][j] = np.sum(np.where(pred_class == j, 1, 0))
    return confusion_matrix


# load csv
train_data = load_csv('train')
test_data = load_csv('test')
x_test = generate_test_data(test_data)

theta_list = []

for i in tqdm(range(10)):
    print(i)
    x_train, y_train = generate_train_data(train_data, i)
    design_mat = build_design_mat(x_train, x_train, 10.)
    theta = optimize_param(design_mat, y_train, 1.)
    theta_list.append(theta)

confusion_matrix = build_confusion_matrix(train_data, x_test, theta_list)
print('confusion matrix')
print(confusion_matrix)
