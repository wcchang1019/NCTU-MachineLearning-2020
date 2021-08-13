import numpy as np


def read_mnist_file(image_file, label_file):
    with open(label_file, 'rb') as f:
        f.read(8)  # magic number, number of items
        labels = np.fromfile(f, dtype=np.uint8)
    with open(image_file, 'rb') as f:
        f.read(16)  # magic number, number of items, number of rows, number of columns
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 28 * 28)
    return images, labels


def discrete_image_pixel(x):
    return x // 128


def get_likelihood(x, label_prob, pixel_prob):
    """
    :param x: 60000 * 784
    :param label_prob: 10 * 1
    :param pixel_prob: 10 * 784
    :return: likelihood: 60000 * 10
    """
    likelihood = np.zeros((x.shape[0], 10))
    for i in range(x.shape[0]):
        for j in range(10):
            likelihood[i, j] = np.prod(pixel_prob[j] ** x[i] * (1 - pixel_prob[j]) ** (1 - x[i])) * label_prob[j]
    return likelihood


def e_step(x, label_prob, pixel_prob):
    """
    :param x: 60000 * 784
    :param label_prob: 10 * 1
    :param pixel_prob: 10 * 784
    :return: ans: 60000 * 10
    """
    ans = get_likelihood(x, label_prob, pixel_prob)
    marginal = np.sum(ans, axis=1).reshape(-1, 1)
    marginal[marginal == 0] = 1
    ans = ans / marginal
    return ans


def m_step(x, w):
    """
    :param x: 60000 * 784
    :param w: 60000 * 10
    :return: new_label_prob: 10 * 1
             new_pixel_prob: 10 * 784
    """
    new_label_prob = np.sum(w, axis=0) / 60000
    marginal = np.sum(w, axis=0).reshape(-1, 1)
    marginal[marginal == 0] = 1
    new_pixel_prob = np.zeros((10, 784))
    for i in range(10):
        for j in range(784):
            new_pixel_prob[i, j] = np.dot(w[:, i].T, x[:, j]) / marginal[i]
    return new_label_prob, new_pixel_prob


def print_imagination(pixel_prob, mapping_label_dict=None):
    for i in range(10):
        if mapping_label_dict is None:
            print(f'class {i}:')
            tmp = pixel_prob[i, :].reshape(28, 28)
        else:
            print(f'labeled class {i}:')
            tmp = pixel_prob[mapping_label_dict[i], :].reshape(28, 28)
        for j in range(28):
            for k in range(28):
                if tmp[j][k] >= 0.5:
                    print('1', end=' ')
                else:
                    print('0', end=' ')
            print()
        print()


def map_label_to_number(x, y, label_prob, pixel_prob):
    ans = get_likelihood(x, label_prob, pixel_prob)
    predict_y = np.argmax(ans, axis=1)
    count_table = np.zeros((10, 10))
    mapping_dict = np.zeros(10).astype(int)
    for i in range(x.shape[0]):
        count_table[y[i]][predict_y[i]] += 1
    for i in range(10):
        a, b = np.where(count_table == count_table.max())
        mapping_dict[a[0]] = int(b[0])
        count_table[a[0], :] = -99999-i
        count_table[:, b[0]] = -99999-i
    return mapping_dict


def initialize_param():
    """
    :return: label_prob: 10 * 1
             pixel_prob: 10 * 784
    """
    label_prob = np.array([0.1] * 10)
    pixel_prob = np.random.rand(10, 28 * 28)
    return label_prob, pixel_prob


def print_confusion_matrix(x, y, label_prob, pixel_prob, mapping_dict):
    ans = get_likelihood(x, label_prob, pixel_prob)
    predict_label = np.array([np.where(mapping_dict == tmp)[0][0] for tmp in np.argmax(ans, axis=1)])
    for label in range(10):
        tp = sum(predict_label[np.where(y == label)[0]] == label)
        fn = sum(predict_label[np.where(y == label)[0]] != label)
        fp = sum(predict_label[np.where(y != label)[0]] == label)
        tn = sum(predict_label[np.where(y != label)[0]] != label)
        print(f'Confusion Matrix {label}:')
        print(f'                Predict number {label} Predict not number {label}')
        print(f'Is number {label}     {tp:^18} {fn:^18}')
        print(f'Is not number {label} {fp:^18} {tn:^18}')
        print(f'Sensitivity (Successfully predict number {label})    : {tp/(tp+fn)}')
        print(f'Specificity (Successfully predict not number {label}): {tn/(fp+tn)}')
        print()
    return sum(predict_label != y)


if __name__ == '__main__':
    X, Y = read_mnist_file('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    X = discrete_image_pixel(X)
    l_prob, p_prob = initialize_param()
    count = 0
    diff = 0
    while count < 100:
        responsibility = e_step(X, l_prob, p_prob)
        new_l_prob, new_p_prob = m_step(X, responsibility)
        new_diff = np.sum(np.abs(p_prob - new_p_prob))
        print_imagination(new_p_prob)
        count += 1
        print(f'No. of Iteration: {count}, Difference: {new_diff}')
        l_prob = new_l_prob
        p_prob = new_p_prob
        if new_diff < 1 and abs(diff - new_diff) < 1:
            break
        diff = new_diff
    mapping_label = map_label_to_number(X, Y, l_prob, p_prob)
    print_imagination(p_prob, mapping_label)
    error_count = print_confusion_matrix(X, Y, l_prob, p_prob, mapping_label)
    print(f'Total iteration to converge: {count}')
    print(f'Total error rate: {error_count / 60000}')
