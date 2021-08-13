import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import imageio
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
plt.rcParams['axes.unicode_minus'] = False


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
        total_distance = cdist(input_image, input_image, 'sqeuclidean')
        while len(center) != k:
            distance = np.min(total_distance[center], axis=0)
            prob = distance / sum(distance)
            center.append(np.random.choice(len(input_image), 1, p=list(prob))[0])
        # Find the the nearest center of each pixel to get label
        cluster_idx = np.argmin(total_distance[center], axis=0)
        return cluster_idx
    else:
        raise ValueError('Unknown method!')


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
    imageio.mimsave(f'gif/spectral-clustering/{gif_name}.gif', images_frame)
    imageio.mimsave(f'gif/spectral-clustering/{gif_name}_tmp.gif', [images_frame[-1], images_frame[-1]])


def get_laplacian(method='ratio_cut'):
    """
    The weighted adjacency matrix $W=(w_{ij})_{i,j=1,...,n}$
    The degree: $d_i=\sum_{j=1}^{n}w_{ij}$
    The degree matrix D: the diagonal matrix with the degrees $d_1,...,d_n$ on the diagonal.
    The unnormalized Laplacian matrix L: $D - W$
    The normalized Laplacian matrix L: $D^{-1/2}LD^{-1/2}$
    (Source: A Tutorial on Spectral Clustering)
    :param method: The cut method used in spectral clustering.
    There are two methods can choose: {'normalized_cut', 'ratio_cut'} (Default: 'ratio_cut')
    :return: The Laplacian matrix
    """
    # unknown cut method
    if method not in ['normalized_cut', 'ratio_cut']:
        raise ValueError('Unknown cut method!')
    degree = np.sum(total_kernel, axis=0)
    degree_square_root = degree ** (-1 / 2)
    degree = np.identity(len(total_kernel)) * degree
    degree_square_root = np.identity(len(total_kernel)) * degree_square_root
    laplacian = degree - total_kernel
    if method == 'normalized_cut':
        laplacian = np.dot(np.dot(degree_square_root, laplacian), degree_square_root)
    return laplacian


def plot_eigen_space(u, cluster_idx, final_center, k, file_name='test'):
    """
    Plot the eigen space of graph Laplacian.
    If k is 2, output the 2D scatter plot.
    If k is 3, output the 3D scatter plot.
    If k is bigger than 3, output the 1D scatter plot of each dimension.
    :param u: the first k eigenvectors(the eigenvectors corresponding to the k smallest eigenvalues)
    :param cluster_idx: the cluster of each pixel
    :param final_center: the center after trained
    :param k: the number of clusters
    :param file_name: output jpg file name
    :return: None
    """
    if k == 2:
        for c_idx in range(k):
            idx = np.where(cluster_idx == c_idx)[0]
            plt.scatter(u[idx, 0], u[idx, 1], c=tuple(final_center[c_idx] / 255))
    elif k == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for c_idx in range(k):
            idx = np.where(cluster_idx == c_idx)[0]
            ax.scatter(u[idx, 0], u[idx, 1], u[idx, 2], c=tuple(final_center[c_idx] / 255))
    else:
        fig, axs = plt.subplots(k, figsize=(15, 15))
        count = 0
        for i in range(k):
            axs[i].set_title(f'eigenspace - the {i} dimension')
            for c_idx in range(k):
                idx = np.where(cluster_idx == c_idx)[0]
                axs[i].scatter(range(count, count+len(idx)), u[idx, i],
                               c=tuple(final_center[c_idx] / 255))
                count = count + len(idx)
    plt.tight_layout()
    plt.savefig(f'gif/spectral-clustering/{file_name}_eigen_space.jpg')


def k_means(input_image, u, gif_name, k=2, max_iter=100, tol=1e-4, method='random'):
    """
    k-means algorithm
    step1: use initialized method to get the cluster of each pixel
    step2: calculate the distance between pixel and center
    step3: update the cluster of each pixel based on the nearest center
    repeat step2 and step3 until convergence or reach the max iteration
    :param input_image: the input image (shape: (10000, 3))
    :param u: the first k eigenvectors of Laplacian
    :param gif_name: the output gif file name
    :param k: the number of clusters (default: 2)
    :param max_iter: the maximum iteration (default: 100)
    :param tol: the tolerance to check convergence (default: 1e-4)
    :param method: the center initialized method (default: 'random')
    :return: the cluster of each pixel
    """
    old_cluster_distance = np.array([0 for _ in range(k)] * len(u)).reshape(-1, len(u))
    cluster_idx = initialize_label(u, k, method)
    center = list()
    for _ in range(k):
        center.append(u[np.random.randint(len(u))])
    cluster_idx_history = [cluster_idx]
    final_center = list()
    final_iter_count = 0
    for iter_count in range(max_iter):
        cluster_distance = list()
        for c_idx in range(k):
            cluster_distance.append(cdist(u, center[c_idx].reshape(1, -1), 'euclidean').flatten())
        cluster_distance = np.array(cluster_distance)
        cluster_idx = np.argmin(cluster_distance, axis=0)
        cluster_idx_history.append(cluster_idx)
        if np.linalg.norm(old_cluster_distance - cluster_distance) < tol:
            final_iter_count = iter_count + 1
            break
        else:
            old_cluster_distance = cluster_distance
        for c_idx in range(k):
            center[c_idx] = np.mean(u[np.where(cluster_idx == c_idx)[0], :], axis=0)
    for c_idx in range(k):
        final_center.append(np.mean(input_image[np.where(cluster_idx == c_idx)[0], :], axis=0))
    make_gif_image(cluster_idx_history, input_image, k, final_center,
                   gif_name=f'{gif_name}_{final_iter_count}')
    plot_eigen_space(u, cluster_idx, final_center, k, file_name=f'{gif_name}_{final_iter_count}')
    return cluster_idx


def spectral_clustering(input_image, k, gif_name,
                        spectral_clustering_method='ratio_cut', initialize_method='random'):
    """
    :param input_image: the input image (shape: (10000, 3))
    :param k: the number of clusters
    :param gif_name: the output gif file name
    :param spectral_clustering_method: The cut method used in spectral clustering.
    There are two methods can choose: {'normalized_cut', 'ratio_cut'} (Default: 'ratio_cut')
    :param initialize_method: There are two initialized method can choose, {'random', 'k-means++'}.
    (Default: 'random')
    :return: None
    """
    laplacian = get_laplacian(spectral_clustering_method)
    # compute eigenvectors
    if os.path.exists(f'{args.image}_{args.cut_method}_{theta1}_{theta2}_a.npy'):
        a = np.load(f'{args.image}_{args.cut_method}_{theta1}_{theta2}_a.npy')
        b = np.load(f'{args.image}_{args.cut_method}_{theta1}_{theta2}_b.npy')
        print('npy file loaded')
    else:
        a, b = np.linalg.eigh(laplacian)
        np.save(f'{args.image}_{args.cut_method}_{theta1}_{theta2}_a', a)
        np.save(f'{args.image}_{args.cut_method}_{theta1}_{theta2}_b', b)
    # find the first k eigenvectors(the eigenvectors corresponding to the k smallest eigenvalues)
    # and drop the first column
    u = b[:, a.argsort()[1:k+1]]
    if spectral_clustering_method == 'normalized_cut':
        # if spectral clustering method is 'normalized_cut', we need to normalize the rows to norm 1
        u = u / cdist(u, np.array([0] * k).reshape(1, -1), 'euclidean')
    return k_means(input_image=input_image, u=u, gif_name=gif_name, k=k, method=initialize_method)


if __name__ == '__main__':
    # use argparse to get all argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default='image1.png',
                        help="input image: image1.png or image2.png (Default: image1.png)")
    parser.add_argument("--initialize_method", default='k-means++',
                        help="initialize method: k-means++ or random (Default: k-means++)")
    parser.add_argument("--cut_method", default='normalized_cut',
                        help="clustering method: normalized_cut or ratio_cut (Default: normalized_cut)")
    parser.add_argument("--theta1", default=0.001, type=float,
                        help="hyper-parameters of spatial information (Default: 0.001)")
    parser.add_argument("--theta2", default=0.001, type=float,
                        help="hyper-parameters of color information (Default: 0.001)")
    parser.add_argument("-k", "--cluster", default=2, type=int,
                        help="the number of clusters (Default: 2)")
    args = parser.parse_args()
    image_name = args.image
    cluster = args.cluster
    theta1 = args.theta1
    theta2 = args.theta2
    image = cv2.imread(image_name).reshape(-1, 3)
    spatial = list()
    for x in range(0, 100):
        for y in range(0, 100):
            spatial.append([x, y])
    spatial = np.array(spatial)
    total_kernel = kernel_function(spatial, spatial, image, image, theta=[theta1, theta2])
    output_name = f'{args.image}_{args.cut_method}_{args.initialize_method}_{cluster}_{theta1}_{theta2}'
    spectral_clustering(image, cluster, output_name,
                        spectral_clustering_method=args.cut_method,
                        initialize_method=args.initialize_method)
