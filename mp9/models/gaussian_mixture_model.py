"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = np.random.uniform(0, 1, size = (self._n_components, self._n_dims))  # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        self._pi = np.random.uniform(0, 1, size = (self._n_components, 1)) # np.array of size (n_components, 1)
        # np.random.dirichlet(tuple(np.random.randint(1, 10, 10)), 1)

        # Initialized with identity.
        sigma_list = []
        for i in range(self._n_components):
            sigma = 1000 * np.eye(self._n_dims)
            sigma_list.append(sigma)
        self._sigma = np.array(sigma_list)  # np.array of size (n_components, n_dims, n_dims)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        mu_index = np.random.choice(x.shape[0],self._n_components)
        self._mu = x[mu_index]

        z_ik = self._e_step(x)
        self._m_step(x, z_ik)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = self.get_posterior(x) + 0.000001

        return z_ik

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        self._pi = np.mean(z_ik, axis=0).reshape(self._n_components, 1)

        mu_list = []
        sigma_list = []
        for comp_index in range(self._n_components):
            z_k = z_ik.T[comp_index].reshape(z_ik.shape[0], 1)
            z_k_sum = np.sum(z_k)
            weight_sum = np.sum(z_k * x, axis=0)
            weight_sum_cov = (z_k * x).T.dot(x)

            mu = weight_sum / z_k_sum
            sigma = weight_sum_cov / z_k_sum
            mu_list.append(mu)
            sigma_list.append(sigma)

        self._mu = np.array(mu_list).reshape(self._n_components, x.shape[1])
        self._sigma = np.array(sigma_list)


    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = []
        for comp_index in range(self._n_components):
            sigma_k = self._sigma[comp_index]
            sigma_det_sqrt = np.sqrt(np.linalg.det(sigma_k))
            sigma_det_verse = np.linalg.inv(sigma_k + self._reg_covar * np.eye(sigma_k.shape[0]))
            a = 1 / ((2 * np.pi) ** (x.shape[1] / 2) * sigma_det_sqrt ** (1 / 2))

            mu_k = self._mu[comp_index]
            b = np.exp(-1 / 2 * np.sum(np.dot((x - mu_k), sigma_det_verse) * (x - mu_k), axis=1))
            z_ik = a * b
            ret.append(z_ik)


        return np.array(ret).T

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        ret = self.get_conditional(x)
        marg_prob = ret.dot(self._pi)

        return marg_prob

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_array = self.get_conditional(x)
        pos1 = z_array * self._pi.T
        pos2 = np.sum(pos1, axis = 1)
        z_ik = pos1 / pos2.reshape(pos2.shape[0],1)

        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.cluster_label_map = []
        z_ik = self.get_posterior(x)
        max_index = np.argmax(z_ik, axis=1)
        for i in range(self._n_components):
            label_index = np.where(max_index == i)
            label_array = y[label_index].astype(np.int)
            if label_array.shape[0] != 0:
                label_bin_count = np.bincount(label_array)
                cluster_label = np.argmax(label_bin_count)
                self.cluster_label_map.append(cluster_label)
            else:
                cluster_label = np.random.choice(np.unique(y).astype(np.int))
                self.cluster_label_map.append(cluster_label)


    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        max_index = np.argmax(z_ik, axis=1)
        y_hat = []
        for index in max_index:
            y_hat.append(self.cluster_label_map[index])

        return np.array(y_hat)
