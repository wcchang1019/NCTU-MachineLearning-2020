import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False


def gaussian_generator(m, s):
    u = np.random.uniform()
    v = np.random.uniform()
    z = (-2 * np.log(u)) ** 0.5 * np.cos(2 * np.pi * v)
    return s ** 0.5 * z + m


def polynomial_basis_linear_model_data_generator(n, a, w):
    x = np.random.uniform(-1, 1)
    y = gaussian_generator(0, a)
    for i in range(n):
        y += w[i] * x ** i
    return x, y


def predictive_distribution():
    b = float(input('b: '))
    n = int(input('n: '))
    a = float(input('a: '))
    a_inv = 1 / a
    w = input('w: ')
    w = [float(x) for x in w.split(',')]
    count = 0
    prior_mu = np.zeros((n, 1))
    prior_var = 1 / b * np.identity(n)
    x_list = []
    y_list = []
    mean_list = []
    variance_list = []
    while True:
        new_data_x, new_data_y = polynomial_basis_linear_model_data_generator(n, a, w)
        count += 1
        x_list.append(new_data_x)
        y_list.append(new_data_y)
        print(f'Add data point ({new_data_x}, {new_data_y}):')
        print()
        x = np.array([new_data_x ** i for i in range(n)]).reshape((1, -1))
        y = new_data_y
        s = np.linalg.inv(prior_var)
        variance = np.linalg.inv(a_inv * np.dot(x.T, x) + s)
        mean = np.dot(variance, a_inv * np.dot(x.T, y) + np.dot(s, prior_mu))
        print('Postirior mean:')
        for tmp_a in range(mean.shape[0]):
            for tmp_b in range(mean.shape[1]):
                print(mean[tmp_a][tmp_b], end=' ')
            print()
        # print(mean)
        print()
        print('Posterior variance:')
        for tmp_a in range(variance.shape[0]):
            for tmp_b in range(variance.shape[1]):
                print(variance[tmp_a][tmp_b], end=' ')
            print()
        # print(variance)
        print()
        predictive_distribution_mean = np.dot(x, mean)[0][0]
        predictive_distribution_variance = (1 / a_inv + np.dot(np.dot(x, variance), x.T))[0][0]
        print(f'Predictive distribution ~ N({predictive_distribution_mean}, {predictive_distribution_variance})')
        print('--------------------------------------------------')
        mean_list.append(mean)
        variance_list.append(variance)
        if np.sum((mean - prior_mu) ** 2) < 1e-6 and np.sum((variance - prior_var) ** 2) < 1e-6 and count > 1000:
            break
        prior_mu = mean
        prior_var = variance
    plot_figure(n, a, w, x_list, y_list, mean_list, variance_list)


def get_pd_list(n, a, mean, variance):
    x = np.linspace(-2, 2, 1000)
    pd_mean_list = []
    pd_upper_bound_list = []
    pd_lower_bound_list = []
    for tmp_x in x:
        xxx = np.array([tmp_x ** i for i in range(n)]).reshape((1, -1))
        predictive_distribution_mean = np.dot(xxx, mean)[0][0]
        pd_mean_list.append(predictive_distribution_mean)
        predictive_distribution_variance = (1 / a + np.dot(np.dot(xxx, variance), xxx.T))[0][0]
        pd_upper_bound_list.append(predictive_distribution_mean + predictive_distribution_variance)
        pd_lower_bound_list.append(predictive_distribution_mean - predictive_distribution_variance)
    return pd_mean_list, pd_upper_bound_list, pd_lower_bound_list


def plot_figure(n, a, w, x_list, y_list, mean_list, variance_list):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    x = np.linspace(-2, 2, 1000)
    y = []
    upper_bound = []
    lower_bound = []
    for tmp_x in x:
        ans = 0
        for i in range(n):
            ans += w[i] * tmp_x ** i
        y.append(ans)
        upper_bound.append(ans + a)
        lower_bound.append(ans - a)
    axs[0, 0].plot(x, y, 'k')
    axs[0, 0].plot(x, upper_bound, 'r')
    axs[0, 0].plot(x, lower_bound, 'r')
    axs[0, 0].set_xlim([-2, 2])
    axs[0, 0].set_ylim([-20, 20])
    axs[0, 0].set_title('Ground truth')
    pd_mean_list, pd_upper_bound_list, pd_lower_bound_list = get_pd_list(n, 1/a, mean_list[-1], variance_list[-1])
    axs[0, 1].plot(x_list, y_list, 'o', markersize=4.5)
    axs[0, 1].plot(x, pd_mean_list, 'k')
    axs[0, 1].plot(x, pd_upper_bound_list, 'r')
    axs[0, 1].plot(x, pd_lower_bound_list, 'r')
    axs[0, 1].set_xlim([-2, 2])
    axs[0, 1].set_ylim([-20, 20])
    axs[0, 1].set_title('Predict result')
    pd_mean_list, pd_upper_bound_list, pd_lower_bound_list = get_pd_list(n, 1/a, mean_list[9], variance_list[9])
    axs[1, 0].plot(x_list[:10], y_list[:10], 'o', markersize=4.5)
    axs[1, 0].plot(x, pd_mean_list, 'k')
    axs[1, 0].plot(x, pd_upper_bound_list, 'r')
    axs[1, 0].plot(x, pd_lower_bound_list, 'r')
    axs[1, 0].set_xlim([-2, 2])
    axs[1, 0].set_ylim([-20, 20])
    axs[1, 0].set_title('After 10 incomes')
    pd_mean_list, pd_upper_bound_list, pd_lower_bound_list = get_pd_list(n, 1/a, mean_list[49], variance_list[49])
    axs[1, 1].plot(x_list[:50], y_list[:50], 'o', markersize=4.5)
    axs[1, 1].plot(x, pd_mean_list, 'k')
    axs[1, 1].plot(x, pd_upper_bound_list, 'r')
    axs[1, 1].plot(x, pd_lower_bound_list, 'r')
    axs[1, 1].set_xlim([-2, 2])
    axs[1, 1].set_ylim([-20, 20])
    axs[1, 1].set_title('After 50 incomes')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    predictive_distribution()
