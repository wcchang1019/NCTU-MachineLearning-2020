import matplotlib.pyplot as plt
import numpy as np


def factorial(n):
    ans = 1
    for i in range(1, n+1):
        ans *= i
    return ans


def gamma_function(x):
    ans = 1
    for i in range(1, x):
        ans *= i
    return ans


def beta_function(a, b):
    return (gamma_function(a) * gamma_function(b)) / gamma_function(a + b)


def beta_distribution(a, b, p):
    return (p ** (a - 1) * (1 - p) ** (b - 1)) / beta_function(a, b)


def binomial_distribution(n, m, p):
    return (factorial(n) / (factorial(n-m) * factorial(m))) * p ** m * (1 - p) ** (n - m)


def plot_prior_likelihood_posterior(a, b, n, m, case_count):
    fig, ax = plt.subplots(1, 3)
    fig.set_figheight(9)
    fig.set_figwidth(16)
    x = np.linspace(0.000001, 0.999999, 250)
    prior = [beta_distribution(a, b, i) for i in x]
    ax[0].plot(x, prior)
    ax[0].set_title('prior')
    likelihood = [binomial_distribution(n, m, i) for i in x]
    ax[1].plot(x, likelihood)
    ax[1].set_title('likelihood')
    new_a = a + m
    new_b = b + n - m
    posterior = [beta_distribution(new_a, new_b, i) for i in x]
    ax[2].plot(x, posterior)
    ax[2].set_title('posterior')
    plt.tight_layout()
    plt.savefig(f'{case_count}')
    plt.show()


def main(initial_a, initial_b):
    a = initial_a
    b = initial_b
    f = open('testfile.txt')
    lines = f.readlines()
    case_count = 1
    for line in lines:
        zero_count = 0
        one_count = 0
        for i in line.strip():
            if i == '0':
                zero_count += 1
            elif i == '1':
                one_count += 1
            else:
                raise ValueError(f'Wrong case - {i}!')
        n = zero_count + one_count
        m = one_count
        plot_prior_likelihood_posterior(a, b, n, m, case_count)
        print(f'case {case_count}: {line.strip()}')
        print(f'Likelihood: {binomial_distribution(n, m, m/n)}')
        print(f'Beta prior:     a = {a}  b = {b}')
        a = a+m
        b = b+n-m
        print(f'Beta posterior: a = {a}  b = {b}')
        case_count += 1
        print()


if __name__ == '__main__':
    input_a = int(input('a: '))
    input_b = int(input('b: '))
    main(input_a, input_b)
