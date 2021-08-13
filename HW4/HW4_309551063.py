import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False


def print_confusion_matrix(predict_label):
    count = int(predict_label.shape[0] / 2)
    tp = sum(predict_label[:count, :] == 0)[0]
    fn = sum(predict_label[:count, :] == 1)[0]
    fp = sum(predict_label[count:, :] == 0)[0]
    tn = sum(predict_label[count:, :] == 1)[0]
    print('Confusion matrix:')
    print('\t\t\t Predict cluster 1 Predict cluster 2')
    print(f'Is cluster 1 {tp:^18} {fn:^18}')
    print(f'Is cluster 2 {fp:^18} {tn:^18}')
    print()
    print(f'Sensitivity (Successfully predict cluster 1): {tp / (tp + fn)}')
    print(f'Specificity (Successfully predict cluster 2): {tn / (tn + fp)}')


def plot_figure(x, nt_label, gd_label):
    fig, axs = plt.subplots(1, 3, figsize=(8, 6))
    count = int(x.shape[0] / 2)
    axs[0].plot(list(x[:count, 0]), list(x[:count, 1]), 'or', markersize=6)
    axs[0].plot(list(x[count:, 0]), list(x[count:, 1]), 'ob', markersize=6)
    axs[0].set_title('Ground truth')
    axs[1].plot(list(x[np.where(gd_label == 0)[0], 0]), list(x[np.where(gd_label == 0)[0], 1]), 'or', markersize=6)
    axs[1].plot(list(x[np.where(gd_label == 1)[0], 0]), list(x[np.where(gd_label == 1)[0], 1]), 'ob', markersize=6)
    axs[1].set_title('Gradient descent')
    axs[2].plot(list(x[np.where(nt_label == 0)[0], 0]), list(x[np.where(nt_label == 0)[0], 1]), 'or', markersize=6)
    axs[2].plot(list(x[np.where(nt_label == 1)[0], 0]), list(x[np.where(nt_label == 1)[0], 1]), 'ob', markersize=6)
    axs[2].set_title('Newton\'s method')
    plt.tight_layout()
    plt.show()


def sigmoid_function(t):
    t[t < -700] = -700
    return 1 / (1 + np.exp(-t))


def gaussian_generator(m, s):
    u = np.random.uniform()
    v = np.random.uniform()
    z = (-2 * np.log(u)) ** 0.5 * np.cos(2 * np.pi * v)
    return s ** 0.5 * z + m


def logistic_regression_gradient_descent(x, y, lr=0.001):
    w = np.zeros((3, 1))
    count = 0
    while count < 5000:
        delta_f = np.dot(x.T, sigmoid_function(np.dot(x, w)) - y)
        new_w = w - lr * delta_f
        if sum(delta_f ** 2) ** 0.5 < 1e-8 or sum((new_w - w) ** 2) ** 0.5 < 1e-8:
            break
        w = new_w
        count += 1
    print('Gradient descent:')
    print()
    print('w:')
    print(w)
    label = sigmoid_function(np.dot(x, w)) > 0.5
    print_confusion_matrix(label)
    return label


def logistic_regression_newtons_method(x, y, lr=0.001):
    w = np.zeros((3, 1))
    count = 0
    while count < 5000:
        d = np.zeros((x.shape[0], x.shape[0]))
        for idx in range(x.shape[0]):
            if -np.dot(x[idx, :], w) > 350:
                d[idx][idx] = 0
            else:
                d[idx][idx] = np.exp(-np.dot(x[idx, :], w)) / ((1 + np.exp(-np.dot(x[idx, :], w))) ** 2)
        h = np.dot(np.dot(x.T, d), x)
        delta_f = np.dot(x.T, sigmoid_function(np.dot(x, w)) - y)
        if not np.linalg.matrix_rank(h) == h.shape[0]:
            new_w = w - lr * delta_f
        else:
            new_w = w - np.dot(np.linalg.inv(h), delta_f)
        if sum(delta_f ** 2) ** 0.5 < 1e-8 or sum((new_w - w) ** 2) ** 0.5 < 1e-8:
            break
        w = new_w
        count += 1
    print('Newton\'s method:')
    print()
    print('w:')
    print(w)
    label = sigmoid_function(np.dot(x, w)) > 0.5
    print_confusion_matrix(label)
    return label


if __name__ == '__main__':
    case = int(input('case:'))
    if case == 1:
        n = 50
        mx1 = 1
        vx1 = 2
        my1 = 1
        vy1 = 2
        mx2 = 10
        vx2 = 2
        my2 = 10
        vy2 = 2
    elif case == 2:
        n = 50
        mx1 = 1
        vx1 = 2
        my1 = 1
        vy1 = 2
        mx2 = 3
        vx2 = 4
        my2 = 3
        vy2 = 4
    else:
        n = int(input('N:'))
        mx1 = float(input('mx_1:'))
        vx1 = float(input('vx_1:'))
        my1 = float(input('my_1:'))
        vy1 = float(input('vy_1:'))
        mx2 = float(input('mx_2:'))
        vx2 = float(input('vx_2:'))
        my2 = float(input('my_2:'))
        vy2 = float(input('vy_2:'))
    data_x = np.ones((2*n, 3))
    data_y = np.zeros((2*n, 1))
    data_y[n:, :] = 1
    for i in range(n):
        data_x[i, 1] = gaussian_generator(mx1, vx1)
        data_x[i, 2] = gaussian_generator(my1, vy1)
    for i in range(n, 2*n):
        data_x[i, 1] = gaussian_generator(mx2, vx2)
        data_x[i, 2] = gaussian_generator(my2, vy2)
    gradient_descent_label = logistic_regression_gradient_descent(data_x, data_y)
    print('----------------------------------------')
    newton_label = logistic_regression_newtons_method(data_x, data_y)
    plot_figure(data_x[:, 1:], newton_label, gradient_descent_label)
