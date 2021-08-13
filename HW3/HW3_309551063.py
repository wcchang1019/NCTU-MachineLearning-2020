import numpy as np


def gaussian_generator(m, s):
    u = np.random.uniform()
    v = np.random.uniform()
    z = (-2 * np.log(u)) ** 0.5 * np.cos(2 * np.pi * v)
    return s ** 0.5 * z + m


def sequential_estimator(m, s):
    print(f'Data point source function: N({m}, {s})')
    print()
    mean = 0
    count = 0
    m2 = 0
    while True:
        new_data = gaussian_generator(m, s)
        print(f'Add data point: {new_data}')
        count += 1
        new_mean = mean + ((new_data - mean) / count)
        new_m2 = m2 + (new_data - mean) * (new_data - new_mean)
        print(f'Mean = {new_mean}   Variance = {new_m2/count}')
        if abs(new_mean - mean) < 1e-5 and abs(new_m2/count - m2/(count - 1)) < 1e-5 and count > 50000:
            break
        mean = new_mean
        m2 = new_m2


if __name__ == '__main__':
    given_m = float(input('m: '))
    given_s = float(input('s: '))
    sequential_estimator(given_m, given_s)
