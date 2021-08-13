#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import argparse
plt.rcParams['axes.unicode_minus'] = False


def plot_scatter(method, perplexity, iter, Y):
    """
    Project all data onto 2D space and mark the data points into different colors respectively
    :param method: which method we use (tsne or ssne)
    :param perplexity: the hyper-parameter perplexity we use
    :param iter: the number of iteration
    :param Y: the high dimensional data project to 2d data
    :return: None
    """
    if not os.path.isdir(method):
        os.mkdir(method)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in np.unique(labels):
        idx = np.where(labels == i)[0]
        plt.scatter(Y[idx, 0], Y[idx, 1], 20, colors[int(i)], label=int(i))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Iteration: {iter}')
    plt.tight_layout()
    plt.savefig(f'{method}/{method}_{perplexity}_{iter}.png')
    plt.close()


def gen_gif(method, perplexity, max_iter):
    """
    In order to show the optimize procedure,
    we use all scatter plots in each iteration to generate the gif
    :param method: which method we use (tsne or ssne)
    :param perplexity: the hyper-parameter perplexity we use
    :param max_iter: the max iteration times
    :return: None
    """
    with imageio.get_writer(f'tsne_ssne_report/{method}_{perplexity}_part2.gif', mode='I') \
            as writer:
        for iter in range(max_iter):
            image = imageio.imread(f'{method}/{method}_{perplexity}_{iter}.png')
            writer.append_data(image)
            if iter == max_iter - 1:
                if os.path.isfile(f'tsne_ssne_report/{method}_{perplexity}_final_result.png'):
                    os.remove(f'tsne_ssne_report/{method}_{perplexity}_final_result.png')
                os.rename(f'{method}/{method}_{perplexity}_{iter}.png',
                          f'tsne_ssne_report/{method}_{perplexity}_final_result.png')
            else:
                os.remove(f'{method}/{method}_{perplexity}_{iter}.png')
    os.rmdir(f'{method}')


def plot_distribution(method, perplexity, P, final_Q):
    """
    Visualize the distribution of pairwise similarities in both high-dimensional space
    and low-dimensional space.
    :param method: which method we use (tsne or ssne)
    :param perplexity: the hyper-parameter perplexity we use
    :param P: probability in high dimension P
    :param final_Q: probability in low dimension Q
    :return: None
    """
    fig, axs = plt.subplots(2)
    # Because the value in P and Q are too small, we use log transform.
    # There are a lot of user defined minimum value in both P and Q,
    # so I delete minimum value to show the distribution normally.
    axs[0].hist(np.log(np.delete(P.flatten(), np.where(P.flatten() == np.min(P))[0])), bins=100)
    axs[0].set_title(f'{method} log P distribution (with perplexity {perplexity})')
    axs[1].hist(np.log(np.delete(final_Q.flatten(),
                                 np.where(final_Q.flatten() == np.min(final_Q))[0])), bins=100)
    axs[1].set_title(f'{method} log Q distribution (with perplexity {perplexity})')
    plt.tight_layout()
    plt.savefig(f'tsne_ssne_report/{method}_{perplexity}_part3.png')


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    let $\beta_i = \frac{1}{2\sigma_i^2}, Si=\sum_{k\neq i}e^{-||x_i-x_k||^2}\beta_i$
    $H(P_i)=-\sum_j p_{j|i}\ln p_{j|i}=\sum_j p_{j|i}(ln(S_i)+{||x_i-x_k||^2}\beta_i)=
    \ln(S_i)+\beta_i\sum_jp_{j|i}||x_i-x_k||^2$
    Given pairwise distance D and beta, use the equation above to get the $H(P_i)$
    :param D: pairwise distance
    :param beta: equal to $\frac{1}{2\sigma_i^2}$
    :return: perplexity, probability in high dimension P
    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    Use binary search to get beta until the perplexity is within tolerance of the goal
    (or until 50 tries have passed).
    :param X: input data
    :param tol: the tolerance between $log(perplexity)$ and $H(P_i)$
    :param perplexity: the hyper-parameter perplexity we use
    :return: probability in high dimension P
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
    Runs PCA on the NxD array X in order to reduce its dimensionality to
    no_dims dimensions.
    Principal Components Analysis(PCA) algorithm
    step1: compute the mean feature vector in high dimension(x)
    step2: find the covariance matrix of mean feature vector
    step3: compute the eigen values and the eigen vectors of covariance matrix
    step4: get the k first largest eigenvectors to become principal components (P)
    :param X: the high dimensional data
    :param no_dims: number of dimensions to keep
    :return: the low dimensional data
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, method='tsne'):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    step1: use pca function to reduce the image data dimension
    step2: use x2p function to compute probability in high dimension p_{i|j}
    based on the given perplexity
    step3: use $p_{ij}=\frac{p_{j|i}+p_{i|j}}{2N}$ to get $p_{ij}$
    step4: use early exaggeration method $p_{ij}=p_{ij}*4$
    (source:https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)
    For each iteration do:
        step5: compute probability in low dimension q_{ij}
        $q_{ij}=\frac{(1+||y_i-y_j||^2)^{-1}}{\sum_{k\neq l}(1+||y_l-y_k||^2)^{-1}}$
        step6: compute the gradient
        $\frac{\delta KL(P||Q)}{\delta y_i}=4\sum_j(p_{ij}-q_{ij})(y_i-y_j)(1+||y_i-y_j||^2)^{-1}$
        step7: use gradient descent with momentum to minimize the KL divergence KL(P||Q)
        $V_t\leftarrow \beta V_{t-1} - \eta \frac{(\delta KL(P||Q))}{\delta Y}*gains$
        $gains=\begin{cases}
        gains+0.2, & \frac{(\delta KL(P||Q))}{\delta Y}*V_{t-1}>0\\
        gains*0.8, & \frac{(\delta KL(P||Q))}{\delta Y}*V_{t-1}<0
        \end{cases}$
        $\beta=\begin{cases}
        0.5, & \text{while iteration < 20} \\
        0.8, & \text{otherwise}
        \end{cases}$
        $\eta=500$
        $Y\leftarrow Y+V_t$

    :param X: input data
    :param no_dims: number of dimensions to keep
    :param initial_dims: number of initial dimensions
    :param perplexity: the hyper-parameter perplexity we use
    :param method: which method we use (tsne or ssne)
    :return: the low dimensional data
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)
    final_Q = np.zeros(P.shape)
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if method == 'ssne':
            num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        elif method == 'tsne':
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            raise NotImplementedError('Not implemented Method')
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        final_Q = Q.copy()
        # Compute gradient
        PQ = P - Q
        if method == 'ssne':
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        elif method == 'tsne':
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
        plot_scatter(method, perplexity, iter, Y)

    gen_gif(method, perplexity, max_iter)
    plot_distribution(method, perplexity, P, final_Q)

    # Return solution
    return Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", default='tsne',
                        help="method: tsne or ssne (Default: tsne)")
    parser.add_argument("-p", "--perplexity", default=20, type=float,
                        help="perplexity (Default: 20)")
    args = parser.parse_args()
    if not os.path.isdir('tsne_ssne_report'):
        os.mkdir('tsne_ssne_report')
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, args.perplexity, method=args.method)
