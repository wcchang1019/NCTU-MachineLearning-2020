import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def matrix_dot(matrix_a, matrix_b):
    assert matrix_a.shape[1] == matrix_b.shape[0], f'matrix shape does not match! {matrix_a.shape}!={matrix_b.shape}'
    ans = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
    for x in range(matrix_a.shape[0]):
        for y in range(matrix_b.shape[1]):
            ans[x, y] = sum([a*b for a, b in zip(matrix_a[x, :].flatten(), matrix_b[:, y].flatten())])
    return ans


def matrix_transpose(matrix):
    ans = np.zeros((matrix.shape[1], matrix.shape[0]))
    for y in range(matrix.shape[1]):
        ans[y, :] = matrix[:, y].flatten()
    return ans


def matrix_multiply_constant(matrix, constant):
    ans = np.zeros((matrix.shape[0], matrix.shape[1]))
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            ans[x, y] = matrix[x, y] * constant
    return ans


def load_data():
    f = open('data.txt')
    lines = f.readlines()
    x_list = list()
    y_list = list()
    for line in lines:
        x = float(line.replace('\n', '').split(',')[0])
        x_list.append(x)
        y = float(line.replace('\n', '').split(',')[1])
        y_list.append(y)
    return x_list, y_list


def plot_fitting_line(n, lse_ans, newton_method_ans, x_data, y_data):
    f = np.arange(min(x_data)-(max(x_data)-min(x_data))*0.5, max(x_data)+(max(x_data)-min(x_data))*0.5, 0.01)
    design_matrix = np.zeros(shape=(len(f), n))
    for i, idx in zip(range(n - 1, -1, -1), range(n)):
        design_matrix[:, idx] = np.array([point ** i for point in f])

    plt.figure(figsize=(12, 7))
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x_data, y_data, 'ro')
    axes[0].plot(f, matrix_dot(design_matrix, lse_ans), 'k')
    axes[0].set_xlim([min(x_data)-(max(x_data)-min(x_data))*0.1, max(x_data)+(max(x_data)-min(x_data))*0.1])
    axes[0].set_ylim([min(y_data) - (max(y_data) - min(y_data)) * 0.1, max(y_data) + (max(y_data) - min(y_data)) * 0.1])
    axes[1].plot(x_data, y_data, 'ro')
    axes[1].plot(f, matrix_dot(design_matrix, newton_method_ans), 'k')
    axes[1].set_xlim([min(x_data)-(max(x_data)-min(x_data))*0.1, max(x_data)+(max(x_data)-min(x_data))*0.1])
    axes[1].set_ylim([min(y_data) - (max(y_data) - min(y_data)) * 0.1, max(y_data) + (max(y_data) - min(y_data)) * 0.1])
    plt.tight_layout()
    plt.show()


def get_upper_triangular_matrix(target_matrix):
    a = np.copy(target_matrix)
    row_len = a.shape[0]
    col_len = a.shape[1]
    count = 1
    for c_idx in range(col_len):
        for r_idx in range(count, row_len):
            times = a[r_idx, c_idx] / a[c_idx, c_idx]
            a[r_idx, :] -= times*a[count-1, :]
        count += 1
    return a


def lu_decomposition(a):
    ans = np.copy(a)
    upper_triangular = get_upper_triangular_matrix(a)
    row_len = upper_triangular.shape[0]
    col_len = upper_triangular.shape[1]
    lower_triangular = np.zeros((row_len, col_len))
    for r_idx in range(row_len):
        count = 0
        for c_idx in range(col_len):
            if c_idx > r_idx:
                continue
            sum_of_dot = 0
            for i in range(col_len):
                if i == count:
                    break
                sum_of_dot += lower_triangular[r_idx, i] * upper_triangular[i, count]
            lower_triangular[r_idx][c_idx] = (ans[r_idx][c_idx] - sum_of_dot) / upper_triangular[count][count]
            count += 1
    return lower_triangular, upper_triangular


def inverse_matrix(target_matrix):
    l, u = lu_decomposition(target_matrix)
    b = np.identity(target_matrix.shape[0])
    ans = np.zeros(target_matrix.shape)
    n = target_matrix.shape[1]
    for b_idx in range(b.shape[0]):
        tmp = np.zeros((n, 1))
        for i in range(n):
            sum_of_dot = 0
            for col_idx in range(n):
                sum_of_dot += l[i][col_idx] * tmp[col_idx][0]
            tmp[i][0] = (b[i][b_idx] - sum_of_dot) / l[i][i]
        for i in range(n-1, -1, -1):
            sum_of_dot = 0
            for col_idx in range(n-1, -1, -1):
                sum_of_dot += u[i][col_idx] * ans[col_idx][b_idx]
            ans[i][b_idx] = (tmp[i][0] - sum_of_dot) / u[i][i]
    return ans


def print_fitting_line(n, method_name, x, error):
    print(f'{method_name}:')
    print('Fitting line: ', end='')
    for i, orders in zip(range(n), range(n-1, -1, -1)):
        if i != 0:
            if x[i][0] > 0:
                print('+', end='')
        print(f'{x[i][0]}', end='')
        if orders != 0:
            print(f'X^{orders}', end='')
        else:
            print()
    print(f'Total error: {error}')


def regularized_least_squares(n, constant_lambda, x_data, y_data):
    design_matrix = np.zeros(shape=(len(x_data), n))
    for i, idx in zip(range(n - 1, -1, -1), range(n)):
        design_matrix[:, idx] = np.array([point ** i for point in x_data])
    target_matrix = \
        matrix_dot(matrix_transpose(design_matrix), design_matrix) \
        + matrix_multiply_constant(np.identity(n), constant_lambda)
    b = matrix_dot(matrix_transpose(design_matrix), np.array(y_data).reshape(-1, 1))
    x = matrix_dot(inverse_matrix(target_matrix), b)
    total_error = 0
    total_error += sum([i ** 2 for i in (matrix_dot(design_matrix, x) - np.array(y_data).reshape(-1, 1)).flatten()])
    return x, total_error


def newton_method(n, x_data, y_data):
    design_matrix = np.zeros(shape=(len(x_data), n))
    b = np.array(y_data).reshape(-1, 1)
    for i, idx in zip(range(n - 1, -1, -1), range(n)):
        design_matrix[:, idx] = np.array([point ** i for point in x_data])
    x0 = np.random.rand(n, 1)
    while True:
        tmp = \
            matrix_multiply_constant(matrix_dot(matrix_dot(matrix_transpose(design_matrix), design_matrix), x0), 2) - \
            matrix_multiply_constant(matrix_dot(matrix_transpose(design_matrix), b), 2)
        x = x0 - matrix_dot(inverse_matrix(matrix_multiply_constant(matrix_dot(matrix_transpose(design_matrix),
                                                                               design_matrix), 2)), tmp)
        if sum([d**2 for d in (x - x0)]) < 1e-10:
            break
        x0 = x
    total_error = sum([i ** 2 for i in (matrix_dot(design_matrix, x) - np.array(y_data).reshape(-1, 1)).flatten()])
    return x, total_error


if __name__ == '__main__':
    N = int(input('n='))
    PARM = float(input('lambda='))
    X_DATA, Y_DATA = load_data()
    lse_weight, lse_error = regularized_least_squares(N, PARM, X_DATA, Y_DATA)
    print_fitting_line(N, 'LSE', lse_weight, lse_error)
    print('')
    newton_method_weight, newton_method_error = newton_method(N, X_DATA, Y_DATA)
    print_fitting_line(N, "Newton's Method", newton_method_weight, newton_method_error)
    plot_fitting_line(N, lse_weight, newton_method_weight, X_DATA, Y_DATA)
