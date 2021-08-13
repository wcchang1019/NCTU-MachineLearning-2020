import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """
    use built-in function to read the input file(file name: input.data)
    :return: np.array data
    """
    x = list()
    y = list()
    with open('input.data', 'r') as f:
        for line in f.readlines():
            x.append(float(line.strip().split(' ')[0]))
            y.append(float(line.strip().split(' ')[1]))
    x = np.array(x)
    y = np.array(y)
    return x, y


def plot_figure(x, y, new_x, mu, variance, theta):
    """
    plot all training data points, the line of mean, and mark the 95% confidence
    :param x: x of input.data
    :param y: y of input.data
    :param new_x: x of test data (range:[-60, 60])
    :param mu: the mean of f (range:[-60, 60])
    :param variance: the variance of f (range:[-60, 60])
    :param theta: the hyper-parameter of kernel
    :return:
    """
    plt.plot(x, y, 'o', color='red')
    # get standard deviation
    error = np.sqrt(variance.diagonal())
    plt.plot(new_x, mu, 'b')
    # mark the 95% confidence
    plt.fill_between(new_x, mu+1.96*error, mu-1.96*error, alpha=0.5)
    plt.title(f'theta: {theta}')
    plt.tight_layout()
    plt.show()


def kernel_function(a, b, theta=None):
    """
    get the rational quadratic kernel:
    $k(x_a,x_b)=\sigma^2(1+\frac{||x_a-x_b||^2}{2\alpha l^2})^{-\alpha}$
    :param a: input matrix
    :param b: input matrix
    :param theta: the parameter in the rational quadratic kernel function
    :return: rational quadratic kernel
    """
    if theta is None:
        theta = np.array([1, 1, 1]).reshape(-1, 1)
    l2_norm = cdist(a.reshape(-1, 1), b.reshape(-1, 1), 'sqeuclidean')
    return (theta[0] ** 2) * (1+l2_norm/(2*theta[1]*(theta[2] ** 2))) ** (-theta[1])


def gaussian_process(x, y, new_x, beta=5, theta=None):
    """
    use the formula in the lecture slide (P.48) to get the mean and variance in Gaussian Process
    $c(x_n,x_m)=k(x_n,x_m)+\beta^{-1}\delta_{nm}$
    $\mu(x^*)=k(x,x^*)^TC^{-1}y$
    $\sigma^2(x^*)=k(x^*,x^*)+\beta^{-1}-k(x,x^*)^TC^{-1}k(x,x^*)$
    :param x: x of input.data
    :param y: y of input.data
    :param new_x: x of test data (range:[-60, 60])
    :param beta: the noisy: N(0, beta^-1) (default: 5)
    :param theta: the parameter in the mix kernel function
    :return: mean, variance
    """
    c = kernel_function(x, x, theta) + (beta ** (-1)) * np.identity(len(x))
    kernel_x_test_x = kernel_function(x, new_x, theta)
    kernel_test_x_test_x = kernel_function(new_x, new_x, theta)
    kernel_x_test_x_t_c_inv = np.dot(kernel_x_test_x.T, np.linalg.inv(c))
    new_mu = np.dot(kernel_x_test_x_t_c_inv, y)
    k_star = kernel_test_x_test_x + (beta ** (-1)) * np.identity(len(kernel_test_x_test_x))
    new_variance = k_star - np.dot(kernel_x_test_x_t_c_inv, kernel_x_test_x)
    return new_mu, new_variance


def minimize_negative_marginal_log_likelihood(x, y, beta=5):
    """
    use scipy.optimize.minimize to minimize negative marginal log likelihood
    :param x: x of input.data
    :param y: y of input.data
    :param beta: the noisy: N(0, beta^-1) (default: 5)
    :return: the optimal hyper-parameter in kernel function
    """
    def negative_log_likelihood(theta):
        """
        the formula in the lecture slide (P.52)
        $-\ln P(y|\theta)=
        \frac{1}{2}\ln|C_{\theta}|+\frac{1}{2}y^TC_{\theta}^{-1}y+\frac{N}{2}\ln (2\pi)$
        :param theta: the hyper-parameter in kernel function
        :return: negative log likelihood
        """
        c = kernel_function(x, x, theta=theta) + (beta ** (-1)) * np.identity(len(x))
        return \
            0.5 * np.log(np.linalg.det(c)) \
            + 0.5 * np.dot(np.dot(y.T, np.linalg.inv(c)), y) \
            + len(x)/2 * np.log(2 * np.pi)
    theta0 = np.array([1, 1, 1])

    def callback(i):
        print(i, negative_log_likelihood(i))
    res = minimize(negative_log_likelihood, theta0, callback=callback,
                   bounds=((-1e-10, 1e+10), (-1e-10, 1e+10), (-1e-10, 1e+10)))
    return res.x


if __name__ == '__main__':
    test_x = np.linspace(-60, 60, 500)
    train_x, train_y = load_data()
    initial_theta = np.array([1, 1, 1])
    gp_mu, gp_variance = gaussian_process(train_x, train_y, test_x, theta=initial_theta)
    plot_figure(train_x, train_y, test_x, gp_mu, gp_variance, initial_theta)
    optimal_theta = minimize_negative_marginal_log_likelihood(train_x, train_y)
    gp_mu, gp_variance = gaussian_process(train_x, train_y, test_x, theta=optimal_theta)
    plot_figure(train_x, train_y, test_x, gp_mu, gp_variance, optimal_theta)
