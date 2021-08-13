import numpy as np
from scipy.spatial.distance import cdist
import cv2
import imageio
import argparse


def kernel_function(spatial_a, spatial_b, color_a, color_b, theta=None):
    """
    Calculate the kernel noticed in the spec.
    $k(x,x')=e^{\gamma_s||S(x)-S(x')||^2}\times e^{\gamma_c||C(x)-C(x')||^2}$
    :param spatial_a: The spatial information of x
    :param spatial_b: The spatial information of x'
    :param color_a: The color information of x
    :param color_b: The color information of x'
    :param theta: The hyper-parameters used in kernel.
                  A list, format: [gamma_s, gamma_c]. (default: [1e-5, 1e-5])
    :return: the mix kernel
    """
    if theta is None:
        theta = np.array([1e-5, 1e-5]).reshape(-1, 1)
    spatial_information = cdist(spatial_a, spatial_b, 'sqeuclidean')
    color_information = cdist(color_a, color_b, 'sqeuclidean')
    return np.exp(- theta[0] * spatial_information) * np.exp(- theta[1] * color_information)


def initialize_label(input_image, k, method='random'):
    """
    Initialize the label of each pixel.
    :param input_image: the input image (shape: (10000, 3))
    :param k: the number of clusters
    :param method: There are two initialized method can choose, {'random', 'k-means++'}.
    (Default: 'random')
    :return: return the initialized label of each pixel
    """
    np.random.seed(666)
    if method == 'random':
        # random to initialize the label
        return np.random.randint(k, size=len(input_image))
    elif method == 'k-means++':
        """
        k-means++ algorithm
        step1: sample a pixel randomly to be center
        step2: calculate the minimum distance D(x) between pixel and the nearest center
        step3: choose a new center based on the probability of $\frac{D(x)^2}{\sum D(x)^2}$
        repeat step2 and step 3 until get enough center
        """
        center = list()
        center.append(np.random.randint(len(input_image)))
        total_distance = kernel_k_means_square_distance_point_point()
        while len(center) != k:
            distance = np.min(total_distance[center], axis=0)
            prob = distance / sum(distance)
            center.append(np.random.choice(len(input_image), 1, p=list(prob))[0])
        # Find the the nearest center of each pixel to get label
        cluster_idx = np.argmin(total_distance[center], axis=0)
        return cluster_idx
    else:
        # Undefined method
        raise ValueError('Unknown initialize label method!')


def kernel_k_means_square_distance_point_point():
    """
    Calculate the square distance based on the kernel.
    $||\phi(x_i)-\phi(x_j)||=
    \phi(x_i)\phi(x_i)-2\phi(x_i)\phi(x_j)+\phi(x_j)\phi(x_j)=k(x_i,x_i)-2k(x_i,x_j)+k(x_j,x_j)$
    :return: the square distance between point and point (shape: (10000, 10000))
    """
    tmp = total_kernel.diagonal().reshape(-1, 1)
    return tmp - 2 * total_kernel + tmp


def kernel_k_means_distance_point_center(target_idx):
    """
    Use the formula in slide P22 to calculate the square distance between point and the center
    $||\phi(x_j)-\mu^{\phi}_k||=
    ||\phi(x_j)-\frac{1}{|C_k|}\sum_{n=1}^{N}\alpha_{kn}\phi(x_n)||
    =k(x_j,x_j)-\frac{2}{|C_k|}\sum_n\alpha_{kn}k(x_j,x_n)+
    \frac{1}{|C_k|^2}\sum_p\sum_q\alpha_{kp}\alpha_{kq}k(x_p,x_q)$
    :param target_idx: the index of the specified center
    :return: the square distance between point and the center
    """
    first_term = total_kernel.diagonal()
    second_term = 2 / len(target_idx)
    alpha = np.zeros(total_kernel.shape[0])
    alpha[target_idx] = 1
    second_term = second_term * np.dot(total_kernel.T, alpha)
    third_term = 1 / (len(target_idx) ** 2)
    third_term = third_term * np.sum(total_kernel[np.ix_(target_idx, target_idx)])
    return first_term - second_term + third_term


def make_gif_image(cluster_idx_history, input_image, k, final_center, gif_name='test'):
    """
    Use the center to colorize the same cluster, and use imageio to make gif file.
    :param cluster_idx_history: the cluster of each pixel in every iteration
    :param input_image: the input image (shape: (10000, 3))
    :param k: the number of clusters
    :param final_center: the center after trained
    :param gif_name: output gif file name
    :return: None
    """
    # use imageio package to generate gif
    images_frame = list()
    for idx in cluster_idx_history:
        image_frame = np.zeros(input_image.shape, dtype=np.uint8)
        for c_idx in range(k):
            # use the center to colorize the same cluster
            image_frame[np.where(idx == c_idx)[0], :] = final_center[c_idx]
        images_frame.append(image_frame.reshape(100, 100, 3))
    imageio.mimsave(f'gif/kernel-k-means/{gif_name}.gif', images_frame)
    # use *_tmp.gif file to show the final result in report
    imageio.mimsave(f'gif/kernel-k-means/{gif_name}_tmp.gif', [images_frame[-1], images_frame[-1]])


def kernel_k_means(input_image, gif_name, k=2, max_iter=100, tol=1e-4, method='random'):
    """
    kernel k-means algorithm
    step1: use initialized method to get the cluster of each pixel
    step2: calculate the distance between pixel and center
    step3: update the cluster of each pixel based on the nearest center
    repeat step2 and step3 until convergence or reach the max iteration
    :param input_image: the input image (shape: (10000, 3))
    :param gif_name: the output gif file name
    :param k: the number of clusters (default: 2)
    :param max_iter: the maximum iteration (default: 100)
    :param tol: the tolerance to check convergence (default: 1e-4)
    :param method: the center initialized method (default: 'random')
    :return: the cluster of each pixel
    """
    old_cluster_distance = np.array(
        [0 for _ in range(k)] * len(input_image)).reshape(-1, len(input_image))
    cluster_idx = initialize_label(input_image, k, method)
    # record the cluster of each pixel in every iteration to output gif
    cluster_idx_history = [cluster_idx]
    final_center = list()
    final_iter_count = 0
    for iter_count in range(max_iter):
        cluster_distance = list()
        for c_idx in range(k):
            cluster_distance.append(
                kernel_k_means_distance_point_center(np.where(cluster_idx == c_idx)[0]))
        cluster_distance = np.array(cluster_distance)
        cluster_idx = np.argmin(cluster_distance, axis=0)
        cluster_idx_history.append(cluster_idx)
        if np.linalg.norm(old_cluster_distance - cluster_distance) < tol:
            final_iter_count = iter_count + 1
            break
        else:
            old_cluster_distance = cluster_distance
    # find the final center
    for c_idx in range(k):
        final_center.append(np.mean(input_image[np.where(cluster_idx == c_idx)[0], :], axis=0))
    make_gif_image(cluster_idx_history, input_image, k, final_center,
                   gif_name=f'{gif_name}_{final_iter_count}')
    return cluster_idx


if __name__ == '__main__':
    # use argparse to get all argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default='image1.png',
                        help="input image: image1.png or image2.png (Default: image1.png)")
    parser.add_argument("-m", "--method", default='k-means++',
                        help="initialize method: k-means++ or random (Default: k-means++)")
    parser.add_argument("--theta1", default=1e-5, type=float,
                        help="hyper-parameters of spatial information (Default: 1e-5)")
    parser.add_argument("--theta2", default=1e-5, type=float,
                        help="hyper-parameters of color information (Default: 1e-5)")
    parser.add_argument("-k", "--cluster", default=2, type=int,
                        help="the number of clusters (Default: 2)")
    args = parser.parse_args()
    initialize_method = args.method
    image_name = args.image
    cluster = args.cluster
    theta1 = args.theta1
    theta2 = args.theta2
    # read image
    image = cv2.imread(image_name).reshape(-1, 3)
    spatial = list()
    for x in range(0, 100):
        for y in range(0, 100):
            spatial.append([x, y])
    spatial = np.array(spatial)
    output_name = f'{image_name}_{cluster}_{initialize_method}_{theta1}_{theta2}'
    # use spatial and image information to get kernel
    total_kernel = kernel_function(spatial, spatial, image, image, theta=[theta1, theta2])
    kernel_k_means(image, gif_name=output_name, k=cluster, method=initialize_method)
