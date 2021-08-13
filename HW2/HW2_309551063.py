import numpy as np
import matplotlib.pyplot as plt
import sys


def read_mnist_file(image_file, label_file):
    with open(label_file, 'rb') as f:
        f.read(8)  # magic number, number of items
        labels = np.fromfile(f, dtype=np.uint8)
    with open(image_file, 'rb') as f:
        f.read(16)  # magic number, number of items, number of rows, number of columns
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 28 * 28)
    return images, labels


def plot_image(image, label):
    plt.imshow(image.reshape(28, 28), cmap='binary')
    plt.title(label)
    plt.show()


def discrete_image_pixel(x):
    return x // 8


def print_label_prediction(total_prob):
    print('Postirior (in log scale):')
    for label in range(10):
        print(f'{label}: {total_prob[label] / sum(total_prob)}')
    prediction = np.argmin([x / sum(total_prob) for x in total_prob])
    return prediction


def naive_bayes_classifier_discrete_mode(x, y, test_x, test_y):
    prior_count = np.zeros(10)
    pixel_count = np.ones((10, 28 * 28, 32))
    for train_idx in range(len(x)):
        prior_count[y[train_idx]] += 1
        for idx in range(len(x[train_idx])):
            pixel_count[y[train_idx]][idx][x[train_idx][idx]] += 1
    error_count = 0
    for test_idx in range(len(test_x)):
        prior_likelihood = []
        for label in range(10):
            prior = np.log((prior_count[label]) / len(x))
            likelihood = 0
            for pixel_idx in range(len(test_x[test_idx])):
                pixel_bin = test_x[test_idx][pixel_idx]
                likelihood += np.log((pixel_count[label][pixel_idx][pixel_bin]) / (prior_count[label]))
            prior_likelihood.append(prior + likelihood)
        prediction = print_label_prediction(prior_likelihood)
        if prediction != test_y[test_idx]:
            error_count += 1
        print(f'Prediction: {prediction}, Ans: {test_y[test_idx]}')
        print()
    print('Imagination of numbers in Bayesian classifier:')
    print()
    for label in range(10):
        print(f'{label}:')
        imagination = np.zeros(28 * 28)
        for i in range(28 * 28):
            if sum(pixel_count[label][i][:16]) < sum(pixel_count[label][i][16:]):
                imagination[i] = 1
        imagination = imagination.reshape(28, 28)
        for a in range(28):
            for b in range(28):
                print(f'{imagination[a][b]:.0f}', end=' ')
            print()
        print()
    print(f'Error rate: {error_count / len(test_x)}')


def log_gaussian_distribution_probability(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma) - 0.5 * ((x-mu) ** 2 / sigma)


def naive_bayes_classifier_continuous_mode(x, y, test_x, test_y):
    pixel_mu = np.zeros((10, 28 * 28))
    pixel_sigma = np.zeros((10, 28 * 28))
    prior_count = np.zeros(10)
    for label in range(10):
        prior_count[label] = len(np.where(y == label)[0])
        target_train_x = x[np.where(y == label)[0]]
        pixel_mu[label, :] = np.mean(target_train_x, axis=0)
        pixel_sigma[label, :] = np.var(target_train_x, axis=0) + 1e-4
    error_count = 0
    for test_idx in range(len(test_x)):
        prior_likelihood = []
        for label in range(10):
            prior = np.log((prior_count[label]) / len(x))
            likelihood = np.sum(log_gaussian_distribution_probability(test_x[test_idx], pixel_mu[label, :],
                                                                      pixel_sigma[label, :]))
            prior_likelihood.append(prior + likelihood)
        prediction = print_label_prediction(prior_likelihood)
        if prediction != test_y[test_idx]:
            error_count += 1
        print(f'Prediction: {prediction}, Ans: {test_y[test_idx]}')
        print()
    for label in range(10):
        print(f'{label}:')
        imagination = pixel_mu[label, :].reshape(28, 28)
        for a in range(28):
            for b in range(28):
                if imagination[a][b] < 128:
                    print('0', end=' ')
                else:
                    print('1', end=' ')
            print()
        print()
    print(f'Error rate: {error_count / len(test_x)}')


if __name__ == '__main__':
    X, Y = read_mnist_file('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    TEST_X, TEST_Y = read_mnist_file('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    if sys.argv[1] == '0':
        X = discrete_image_pixel(X)
        TEST_X = discrete_image_pixel(TEST_X)
        naive_bayes_classifier_discrete_mode(X, Y, TEST_X, TEST_Y)
    elif sys.argv[1] == '1':
        naive_bayes_classifier_continuous_mode(X, Y, TEST_X, TEST_Y)
    else:
        raise ValueError('Unknown toggle option!(0: discrete mode, 1: continuous mode)')
