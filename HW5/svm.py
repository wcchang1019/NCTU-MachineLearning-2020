import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist


def read_csv(file_name):
    """
    use built-in function to read the input file
    :param file_name: csv file name
    :return: np.array data
    """
    ans = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            ans.append([float(x) for x in line.strip().split(',')])
    return np.array(ans)


def load_data():
    """
    load the train and test csv data
    :return: x_train, y_train, x_test, y_test
    """
    x_test = read_csv('X_test.csv')
    x_train = read_csv('X_train.csv')
    y_test = read_csv('Y_test.csv').reshape(-1, )
    y_train = read_csv('Y_train.csv').reshape(-1, )
    return x_train, y_train, x_test, y_test


def train_svm(x_train, y_train, x_test, y_test, kernel_name='linear', cross_validation=0,
              gamma=None, cost=None):
    """
    :param x_train: the input of training data
    :param y_train: the label of training data
    :param x_test: the input of testing data
    :param y_test: the label of testing data
    :param kernel_name: specify the kernel used in libsvm
    There are three type: 'linear','polynomial', 'radial' (default: linear)
    :param cross_validation: the number of fold used in cross validation (default: 0)
    :param gamma: specify the gamma in libsvm (default: None)
    :param cost: specify the cost in libsvm (default: None)
    :return: the accuracy in training data
    """
    # mapping the kernel to libsvm kernel_type
    kernel_dict = {'linear': 0, 'polynomial': 1, 'radial': 2}
    prob = svm_problem(y_train, x_train)
    print(f'Training svm with {kernel_name} kernel...')
    # setting parameter in libsvm, -q means quiet mode(no output),-t means the kernel_type,
    # -g means gamma, -c means cost, -v means how many fold used in the cross validation
    if not cross_validation:
        par = f'-q -t {kernel_dict[kernel_name]}'
        if gamma is not None:
            par += f' -g {gamma}'
        if cost is not None:
            par += f' -c {cost}'
        param = svm_parameter(par)
    else:
        if gamma is not None and cost is not None:
            param = svm_parameter(f'-q -t {kernel_dict[kernel_name]} -v {cross_validation} '
                                  f'-g {gamma} -c {cost}')
        else:
            raise ValueError('gamma and cost must be specified in cross validation.')
    # training the svm
    m = svm_train(prob, param)
    print('Done')
    # use the trained model to predict the testing data if not in cross validation
    if not cross_validation:
        print(f'Testing svm with {kernel_name} kernel...')
        # This function would print the accuracy in testing data
        svm_predict(y_test, x_test, m)
    return m


def question1(x_train, y_train, x_test, y_test):
    """
    In part1, use 'linear', 'polynomial', 'RBF' kernel to train svm and compare the accuracy
    :param x_train: the input of training data
    :param y_train: the label of training data
    :param x_test: the input of testing data
    :param y_test: the label of testing data
    :return: None
    """
    for k in ['linear', 'polynomial', 'radial']:
        train_svm(x_train, y_train, x_test, y_test, k)


def grid_search(x_train, y_train, x_test, y_test, gamma, cost):
    """
    Given a list of gamma and a list of cost to create some hyper-parameters,
    then use grid search and cross validation to find the best hyper-parameter
    in RBF kernel(gamma and cost)
    :param x_train: the input of training data
    :param y_train: the label of training data
    :param x_test: the input of testing data
    :param y_test: the label of testing data
    :param gamma: a list of gamma
    :param cost: a list of cost
    :return: a list of best hyper-parameter ([best gamma, best cost])
    """
    # record the best accuracy and its parameter
    best_acc = 0
    best_param = [gamma[0], cost[0]]
    acc_arr = np.zeros((len(gamma), len(cost)))
    for i, g in enumerate(gamma):
        for j, c in enumerate(cost):
            # grid search all given parameter
            print(f"gamma: {g}, cost: {c}")
            # use rbf kernel and 3-fold cross validation given gamma and cost
            result = train_svm(x_train, y_train, x_test, y_test,
                               kernel_name='radial', cross_validation=3, gamma=g, cost=c)
            # if this parameter's accuracy outperform before, record it
            if result > best_acc:
                best_acc = result
                best_param = [g, c]
            acc_arr[i, j] = result
    print(best_acc, best_param)
    return best_param


def question2(x_train, y_train, x_test, y_test):
    """
    Use the cross-validation and grid-search method which is recommended in the libsvm user guide.
    First, use the coarse grid search to find the hyper-parameter region.
    And then use this region to find a better solution.
    :param x_train: the input of training data
    :param y_train: the label of training data
    :param x_test: the input of testing data
    :param y_test: the label of testing data
    :return: None
    """
    # First grid search
    print('First grid search, gamma range:{2^-15, 2^-13, ..., 2^3}, '
          'cost range:{2^-5, 2^-3, ..., 2^15}')
    gamma = []
    for i in range(10):
        gamma.append(2 ** (i * 2 - 15))
    cost = []
    for i in range(11):
        cost.append(2 ** (i * 2 - 5))
    best_param = grid_search(x_train, y_train, x_test, y_test, gamma, cost)
    print(f"{best_param}")
    # because the best gamma is 2^-5, and the best cost is 2^3,
    # so use smaller region to do second grid search
    print('Second grid search, gamma range:{2^-7.5, 2^-7, ..., 2^-3}, '
          'cost range:{2^0.5, 2^1, ..., 2^5.5}')
    gamma = []
    for i in range(10):
        gamma.append(2 ** (i * 0.5 - 7.5))
    cost = []
    for i in range(11):
        cost.append(2 ** (i * 0.5 + 0.5))
    best_param = grid_search(x_train, y_train, x_test, y_test, gamma, cost)
    print(f"{best_param}")
    # use the best gamma and best cost to train the model and get the testing performance
    train_svm(x_train, y_train, x_test, y_test, kernel_name='radial',
              gamma=best_param[0], cost=best_param[1])


def mix_kernel_function(a, b, theta=None):
    """
    get the linear kernel + RBF kernel function: theta_0*e^{(-theta_1*|a-b|^2)}+theta_2*(a*b')
    :param a: input matrix
    :param b: input matrix
    :param theta: the parameter in the mix kernel function
    :return: linear kernel + RBF kernel function
    """
    if theta is None:
        theta = np.array([1, 1, 1]).reshape(-1, 1)
    rbf = cdist(a, b, 'sqeuclidean')
    linear = np.dot(a, b.T)
    # if we want to use precomputed kernel, the first column must be the index start with 1.
    # hence we use np.hstack to combine index and the mix kernel
    return np.hstack((np.array(range(1, len(a)+1)).reshape(-1, 1),
                      theta[0]*np.exp(-theta[1]*rbf)+theta[2]*linear))


def question3(x_train, y_train, x_test, y_test):
    """
    Use precomputed kernel(linear kernel + RBF kernel) to train the svm
    :param x_train: the input of training data
    :param y_train: the label of training data
    :param x_test: the input of testing data
    :param y_test: the label of testing data
    :return: None
    """
    # get the mix kernel in training
    tmp = mix_kernel_function(x_train, x_train)
    # isKernel=True must be set for precomputed kernel
    prob = svm_problem(y_train, tmp, isKernel=True)
    # -t 4 means precomputed kernel
    par = f'-q -t 4'
    param = svm_parameter(par)
    m = svm_train(prob, param)
    # get the mix kernel in testing
    tmp = mix_kernel_function(x_test, x_train)
    svm_predict(y_test, tmp, m)


if __name__ == '__main__':
    X, Y, TEST_X, TEST_Y = load_data()
    question1(X, Y, TEST_X, TEST_Y)
    question2(X, Y, TEST_X, TEST_Y)
    question3(X, Y, TEST_X, TEST_Y)
