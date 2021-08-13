import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def rbf_kernel(a, b, theta=1e-8):
    """
    Calculate the rbf kernel $e^{-theta*|a-b|^2}$
    :param a: input matrix
    :param b: input matrix
    :param theta: the hyper-parameter in rbf kernel
    :return: the radial basis kernel
    """
    distance = cdist(a, b, 'sqeuclidean')
    return np.exp(-theta*distance)


def polynomial_kernel(a, b, gamma=1e-10, coef=1e-3, degree=3):
    """
    Calculate the polynomial kernel $(gamma*a*b'+coef)^{degree}$
    :param a: input matrix
    :param b: input matrix
    :param gamma: the hyper-parameter in polynomial kernel
    :param coef: the hyper-parameter in polynomial kernel
    :param degree: the hyper-parameter in polynomial kernel
    :return: the polynomial kernel
    """
    return (gamma * np.dot(a, b.T) + coef) ** degree


def linear_kernel(a, b):
    """
    Calculate the linear kernel $a*b'$
    :param a: input matrix
    :param b: input matrix
    :return: the linear kernel
    """
    return np.dot(a, b.T)


def sigmoid_kernel(a, b, gamma=1e-13, coef=1e-3):
    """
    Calculate the sigmoid kernel $tanh(gamma*a*b'+coef)$
    :param a: input matrix
    :param b: input matrix
    :param gamma: the hyper-parameter in sigmoid kernel
    :param coef: the hyper-parameter in sigmoid kernel
    :return: the sigmoid kernel
    """
    return np.tanh(gamma * np.dot(a, b.T) + coef)


def load_data(folder_name):
    """
    Load face data, resize them to specific size and load subject labels from the folder
    :param folder_name: the data folder
    :return: face data (shape: (#data, W*H)), subject labels (shape: (#data, ))
    """
    image_data = list()
    labels = list()
    for file_name in os.listdir(folder_name):
        img = np.array(Image.open(f'{folder_name}/{file_name}').resize((W, H), Image.ANTIALIAS))
        image_data.append(img)
        labels.append(int(file_name.split('.')[0][7:9]))
    return np.array(image_data).reshape(-1, W * H), np.array(labels)


def pca(input_x, n_components=25):
    """
    Principal Components Analysis(PCA) algorithm
    step1: compute the mean feature vector in high dimension(x)
    step2: find the covariance matrix of mean feature vector
    step3: compute the eigen values and the eigen vectors of covariance matrix
    step4: get the k first largest eigenvectors to become principal components (P)
    :param input_x: the training face data
    :param n_components: number of dimensions to keep
    :return: the principal components (P)
    """
    mean = np.mean(input_x, axis=0)
    x_minus_mu = input_x - mean
    s = np.dot(x_minus_mu.T, x_minus_mu) / len(input_x)
    # because s is a symmetric matrix, we can use np.linalg.eigh to speed up
    eigen_values, eigen_vectors = np.linalg.eigh(s)
    principal_components = eigen_vectors[:, np.argsort(eigen_values)[::-1][:n_components]]
    return principal_components


def kernel_pca(input_x, kernel_function, n_components=25):
    """
    kernel PCA algorithm
    step1: compute the kernel
    step2: compute $K^C=K-1_NK-K1_N+1_NK1_N$ (from lecture slides P128)
    step3: compute the eigen values and the eigen vectors of $K^C$
    step4: sort the eigen values and the eigen vectors and normalize eigen vectors
    step5: get the k first largest eigenvectors to become principal components (P)
    :param input_x: the training face data
    :param kernel_function: which kernel function used in kernel pca
    :param n_components: number of dimensions to keep
    :return: the principal components (P), the mean of training kernel in each column,
             the mean of total training kernel, the projection in low dimension of training data
    """
    kernel = kernel_function(input_x, input_x)
    one_n = np.ones(kernel.shape) / len(kernel)
    kc = kernel - np.dot(one_n, kernel) - np.dot(kernel, one_n) + np.dot(np.dot(one_n, kernel), one_n)
    # because kc is a symmetric matrix, we can use np.linalg.eigh to speed up
    lambdas, alphas = np.linalg.eigh(kc)
    index = range(lambdas.shape[0])[::-1]
    lambdas = lambdas[index]
    alphas = alphas[:, index]
    index = lambdas > 0
    lambdas = lambdas[index]
    alphas = alphas[:, index]
    alphas = alphas / np.sqrt(lambdas)
    principal_components = alphas[:, :n_components]
    return principal_components, np.mean(kernel, axis=1), np.mean(kernel), np.dot(kc, principal_components)


def plot_face(projection_matrix, title):
    """
    Use projection matrix to plot the first 25 faces
    :param projection_matrix: the projection matrix in PCA or LDA
    :param title: the title of the figure
    :return: None
    """
    fig, axs = plt.subplots(5, 5, figsize=(17, 17))
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(projection_matrix[:, i * 5 + j].reshape((H, W)), cmap='gray')
            axs[i, j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title, fontsize=50)
    plt.savefig(f'PCA_LDA_report/{title}.jpg')


def plot_reconstruction(reconstruction, title):
    """
    Use reconstruction to plot the random 10 images
    :param reconstruction: the reconstruction in PCA or LDA
    :param title: the title of the figure
    :return: None
    """
    np.random.seed(666)
    idx = np.random.randint(len(reconstruction), size=10)
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))
    for i in range(4):
        for j in range(3):
            if i * 3 + j < len(idx):
                axs[i, j].imshow(reconstruction[idx[i*3+j], :].reshape((H, W)), cmap='gray')
            axs[i, j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title, fontsize=50)
    plt.savefig(f'PCA_LDA_report/{title}.jpg')


def lda(input_x, input_y, n_components=25):
    """
    Linear Discriminative Analysis(LDA) algorithm
    ($N_c$ is the number of samples of $c$ class)
    step1: compute mean vector for each class: $m_c=\frac{\sum^{N_c}_{i=1}x_i}{N_c}$
    step2: compute within-class scatter: $S_w=\sum^C_{c=1}\sum^{N_c}_{j=1}(x_j-m_c)(x_j-m_c)^T$
    step3: compute total mean vector: $m=\frac{\sum^N_{i=1}x_i}{N}$
    step4: compute between-class scatter: $S_b=\sum^C_{c=1}N_c(m_c-m)(m_c-m)^T$
    step5: compute the eigen values and the eigen vectors of $S_w^{-1}S_b$
    step6: get the k first largest eigenvectors to become projection matrix
    :param input_x: the training face data
    :param input_y: the labels of training data
    :param n_components: number of dimensions to keep
    :return: the projection matrix
    """
    # if os.path.isfile('lda_projection.npy'):
    #     return np.load('lda_projection.npy')
    mc = list()
    for subject in np.unique(input_y):
        mc.append(np.mean(input_x[np.where(np.array(input_y) == subject)[0]], axis=0))
    mc = np.array(mc)
    total_mc = np.mean(input_x, axis=0)
    sw = np.zeros((input_x.shape[1], input_x.shape[1]))
    for subject in np.unique(input_y):
        for j in np.where(np.array(input_y) == subject)[0]:
            tmp = (input_x[j] - mc[subject-1]).reshape(-1, 1)
            sw += np.dot(tmp, tmp.T)
    sb = np.zeros((input_x.shape[1], input_x.shape[1]))
    for subject in np.unique(input_y):
        idx = np.where(np.array(input_y) == subject)[0]
        tmp = (mc[subject-1] - total_mc).reshape(-1, 1)
        sb += len(idx) * np.dot(tmp, tmp.T)
    # because the number of data is lower than the dimension, sw becomes un-invertible
    # so use pseudo inverse
    sw_pinv = np.linalg.pinv(sw, rcond=1e-10)
    w = np.dot(sw_pinv, sb)
    # because w is not a symmetric matrix, we can't use np.linalg.eigh
    eigen_values, eigen_vectors = np.linalg.eig(w)
    np.save('lda_projection', eigen_vectors[:, np.argsort(eigen_values)[::-1][:25]].real)
    return eigen_vectors[:, np.argsort(eigen_values)[::-1][:n_components]].real


def kernel_lda(input_x, input_y, kernel_function, n_components=25):
    """
    kernel LDA algorithm (source: https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis)
    step1: compute $(M_*)_j=\frac{1}{l}\sum_{k=1}^{l}k(x_j, x_k)$
           where $l$ is the number of total data
    step2: compute $M=\sum_{j=1}^c l_j(M_j-M_*)(M_j-M_*)^T$
           where $l_j$ is the number of examples of class $C_j$
    step3: compute $N=\sum_{j=1}^c K_j(I-1_{l_j})K_j^T$
           where $1_{l_j}$ is the matrix with all entries equal to $\frac{1}{l_j}$
    step4: compute the eigen values and the eigen vectors of $N^{-1}M$
    step5: get the k first largest eigenvectors to become projection matrix
    :param input_x: the training face data
    :param input_y: the labels of training data
    :param kernel_function: which kernel function used in kernel LDA
    :param n_components: number of dimensions to keep
    :return: the projection matrix, the projection in low dimension of training data
    """
    kernel = kernel_function(input_x, input_x)
    mc = list()
    for subject in np.unique(input_y):
        mc.append(np.mean(kernel[np.where(input_y == subject)[0], :], axis=0))
    mc = np.mean(kernel, axis=1)
    M = np.zeros((135, 135))
    for subject in np.unique(input_y):
        l_j = len(np.where(input_y == subject)[0])
        m_j = np.mean(kernel[np.where(input_y == subject)[0], :], axis=0)
        tmp = (m_j - mc).reshape(-1, 1)
        M += l_j * np.dot(tmp, tmp.T)
    N = np.zeros((135, 135))
    for subject in np.unique(input_y):
        k_j = kernel[:, np.where(input_y == subject)[0]]
        tmp = np.identity(k_j.shape[1]) / len(np.where(input_y == subject)[0])
        N += np.dot(np.dot(k_j, tmp), k_j.T)
    # in practice, N is usually singular and so a multiple of the identity is added to it
    epsilon = 0.00001
    N_inv = np.linalg.inv(N+epsilon*np.eye(N.shape[0]))
    w = np.dot(N_inv, M)
    # because w is not a symmetric matrix, we can't use np.linalg.eigh
    eigen_values, eigen_vectors = np.linalg.eig(w)
    pc = eigen_vectors[:, np.argsort(eigen_values)[::-1][:n_components]].real
    return pc, np.dot(kernel, pc)


def knn(k_neighbors, input_x, input_y, testing_x):
    """
    k-nearest neighbors algorithm (KNN)
    for each testing data do:
        step1: compute the distance between this testing data and training data
        step2: assign the label which is most frequent among
        the k training samples nearest to the testing data
    :param k_neighbors: number of neighbors
    :param input_x: the training face data
    :param input_y: the labels of training data
    :param testing_x: the testing face data
    :return: the predicted labels of testing data
    """
    prediction = list()
    for testing_data in testing_x:
        distance = cdist(testing_data.reshape(1, -1), input_x, 'euclidean').flatten()
        nearest_idx = np.argsort(distance)[:k_neighbors]
        u, count = np.unique(np.array(input_y)[nearest_idx], return_counts=True)
        count_sort_ind = np.argsort(-count)
        prediction.append(u[count_sort_ind[0]])
    return np.array(prediction)


def part1(projection_matrix, input_x, title, method):
    """
    Use PCA and LDA to show the first 25 eigen_faces and fisher_faces,
    and randomly pick 10 images to show their reconstruction.
    P: projection matrix
    reconstruction: $(x-\overline x)PP^T + \overline x$
    :param projection_matrix: the projection matrix in PCA or LDA
    :param input_x: the training face data
    :param title: the title in the figure of the first 25 faces
                  ('eigen_faces' or 'fisher_faces')
    :param method: which method we use ('PCA' or 'LDA')
    :return: None
    """
    mean = np.mean(input_x, axis=0)
    x_minus_mu = input_x - mean
    plot_face(projection_matrix, title)
    reconstruction = np.dot(np.dot(x_minus_mu, projection_matrix), projection_matrix.T) + mean
    plot_reconstruction(reconstruction, f'{method} reconstruction')
    plot_reconstruction(input_x, 'original images')


def part2(projection_matrix, input_x, input_y, testing_x, testing_y, method, k_neighbors=3):
    """
    use KNN and PCA or LDA to do the face recognition, and compute the performance.
    P: projection matrix
    the projection in low dimension: $(x-\overline x)P$
    step1: use projection matrix to compute the projection in low dimension
           (training data and testing data)
    step2: use both projections as input of KNN to do the face recognition
    step3: output the accuracy
    :param projection_matrix: the projection matrix in PCA or LDA
    :param input_x: the training face data
    :param input_y: the labels of training data
    :param testing_x: the testing face data
    :param testing_y: the labels of testing data
    :param method: which method we use ('PCA' or 'LDA')
    :param k_neighbors: the number of neighbors in KNN
    :return: None
    """
    mean = np.mean(input_x, axis=0)
    testing_with_zero_mean = testing_x - mean
    projection = np.dot(testing_with_zero_mean, projection_matrix)
    trained_projection = np.dot(input_x - mean, projection_matrix)
    prediction = knn(k_neighbors, trained_projection, input_y, projection)
    print(f'{method} accuracy: {np.mean(prediction == testing_y) * 100}%')


def part3(input_x, input_y, testing_x, testing_y, k_neighbors=3, method='kernel_pca'):
    """
    use KNN and kernel PCA to do the face recognition, and compute the performance.
    step1: use kernel_pca to get the principal components (P),
           the mean of training kernel in each column, the mean of total training kernel,
           the projection in low dimension of training data
    step2: use testing data and training data to compute the kernel
    step3: use the equation in lecture slides(P128) to compute $K^C$ in testing data kernel
    $K^C_{ij}=K_{ij}-\frac{1}{N}\sum_lK_{il}-\frac{1}{N}\sum_kK_{kj}+\frac{1}{N^2}\sum_k\sum_lK_{kl}$
    step4: use P and the $K^C$ in testing data kernel to get the low dimension of testing data
    step5: use both(training and testing) projections as input of KNN to do the face recognition
    step6: output the accuracy

    use KNN and kernel LDA to do the face recognition, and compute the performance.
    step1: use kernel_lda to get the projection matrix, the projection in low dimension of training data
    step2: use testing data and training data to compute the kernel
    step3: use projection matrix and the testing data kernel to get the low dimension of testing data
    step4: use both(training and testing) projections as input of KNN to do the face recognition
    step5: output the accuracy

    :param input_x: the training face data
    :param input_y: the labels of training data
    :param testing_x: the testing face data
    :param testing_y: the labels of testing data
    :param k_neighbors: the number of neighbors in KNN
    :param method: which method we use ('kernel_pca' or 'kernel_lda')
    :return: None
    """
    for kernel_function in [rbf_kernel, polynomial_kernel, linear_kernel, sigmoid_kernel]:
        if method == 'kernel_pca':
            kernel_pc, kernel_mean, total_mean, training_projection = kernel_pca(input_x, kernel_function)
            testing_kernel = kernel_function(testing_x, input_x)
            testing_kernel = testing_kernel - kernel_mean - np.mean(testing_kernel, axis=1)[:, np.newaxis] + total_mean
            testing_projection = np.dot(testing_kernel, kernel_pc)
            prediction = knn(k_neighbors, training_projection, input_y, testing_projection)
            print(f'PCA with {kernel_function.__name__} accuracy: {np.mean(prediction == testing_y) * 100:.2f}%')
        elif method == 'kernel_lda':
            kernel_pc, training_projection = kernel_lda(input_x, input_y, kernel_function)
            testing_kernel = kernel_function(testing_x, input_x)
            testing_projection = np.dot(testing_kernel, kernel_pc)
            prediction = knn(k_neighbors, training_projection, input_y, testing_projection)
            print(f'LDA with {kernel_function.__name__} accuracy: {np.mean(prediction == testing_y) * 100:.2f}%')
        else:
            raise ValueError('Not Implemented method!')


def PCA_part():
    pc = pca(train_x)
    part1(pc, train_x, 'eigen_faces', 'PCA')
    part2(pc, train_x, train_label, test_x, test_label, 'PCA', k)
    part3(train_x, train_label, test_x, test_label, k, method='kernel_pca')


def LDA_part():
    projection_matrix = lda(train_x, train_label)
    part1(projection_matrix, train_x, 'fisher_faces', 'LDA')
    part2(projection_matrix, train_x, train_label, test_x, test_label, 'LDA', k)
    part3(train_x, train_label, test_x, test_label, k, method='kernel_lda')


if __name__ == '__main__':
    if not os.path.isdir('PCA_LDA_report'):
        os.mkdir('PCA_LDA_report')
    # resize the images
    W, H = 77, 65
    for k in [5, 10, 15, 20, 25]:
        print(f'k: {k}')
        train_x, train_label = load_data('Yale_Face_Database/Training')
        test_x, test_label = load_data('Yale_Face_Database/Testing')
        PCA_part()
        LDA_part()
        plt.close('all')
