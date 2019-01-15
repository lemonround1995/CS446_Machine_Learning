"""Variation autoencoder."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers
from tensorflow.contrib.slim import fully_connected


class VariationalAutoencoder(object):
    """Varational Autoencoder.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a VAE. (**Do not change this function**)

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def _sample_z(self, z_mean, z_log_var):
        """Samples z using reparametrization trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)
        Returns:
            z (tf.Tensor): Random sampled z of dimension (None, _nlatent)
        """

        z = None
        ####### Implementation Here ######
        # epsilon = tf.random_normal(shape=tf.shape(self.z_log_var), mean=0, stddev=1, dtype = tf.float32)
        epsilon = tf.contrib.distributions.MultivariateNormalDiag(loc=np.zeros(self._nlatent, dtype = np.float32))\
                                         .sample(sample_shape=tf.shape(z_mean)[0])
        z = z_mean + tf.sqrt(tf.exp(z_log_var)) * epsilon
        # z = z_mean + tf.matmul(tf.exp(z_log_var)**1/2, epsilon)
        return z

    def _encoder(self, x):
        """Encoder block of the network.

        Builds a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes, and outputs two branches each with _nlatent nodes
        representing z_mean and z_log_var. Network illustrated below:

                             |-> _nlatent (z_mean)
        Input --> 100 --> 50 -
                             |-> _nlatent (z_log_var)

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, _ndims).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension
                (None, _nlatent).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, _nlatent).
        """
        z_mean = None
        z_log_var = None
        ####### Implementation Here ######
        layer1 = fully_connected(x, 100, activation_fn= tf.nn.softplus)
        layer2 = fully_connected(layer1, 50, activation_fn = tf.nn.softplus)
        z_mean = fully_connected(layer2, self._nlatent,activation_fn = None)
        z_log_var = fully_connected(layer2, self._nlatent, activation_fn= None) # should activation_fn = None for output layer?

        return z_mean, z_log_var

    def _decoder(self, z):
        """From a sampled z, decode back into image.

        Builds a three layer network of fully connected layers,
        with 50, 100, _ndims nodes.

        z (_nlatent) --> 50 --> 100 --> _ndims.

        Use activation of tf.nn.softplus for hidden layers.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, _nlatent).
        Returns:
            f(tf.Tensor): Decoded features, tensor of dimension (None, _ndims).
        """

        f = None # This is the predicted value
        ####### Implementation Here ######
        layer1 = fully_connected(z, 50, activation_fn = tf.nn.softplus)
        layer2 = fully_connected(layer1, 100, activation_fn= tf.nn.softplus)
        f = fully_connected(layer2, self._ndims, activation_fn = tf.nn.sigmoid)

        return f

    def _latent_loss(self, z_mean, z_log_var):
        """Constructs the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, _nlatent)
            z_log_var(tf.Tensor): Tensor of dimension (None, _nlatent)

        Returns:
            latent_loss(tf.Tensor): A scalar Tensor of dimension ()
                containing the latent loss.
        """
        latent_loss = None
        ####### Implementation Here ######
        # z_var = tf.exp(z_log_var)
        # tr_z = tf.cast(tf.reduce_sum(z_var), tf.float32)
        # mul_mean = tf.cast(tf.reduce_sum(z_mean**2), tf.float32)
        # d_term = tf.cast(tf.shape(z_mean)[0] * self._nlatent, tf.float32)
        # log_prod = tf.cast(tf.log(tf.reduce_prod(z_var)), tf.float32)
        # sample_num = tf.cast(tf.shape(z_mean)[0], tf.float32)
        # latent_loss = 0.5 * (tr_z + mul_mean - d_term - log_prod) / sample_num
        # ========================================================================
        # p_z = tf.contrib.distributions.Normal(loc= np.zeros(self._nlatent, dtype = np.float32),\
                                              # scale = np.ones(self._nlatent, dtype = np.float32))
        # q_z = tf.contrib.distributions.Normal(loc = z_mean, scale = tf.exp(0.5 * z_log_var))
        # latent_loss = -tf.reduce_mean(tf.reduce_sum(tf.contrib.distributions.kl_divergence(q_z, p_z), axis=1))
        latent_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis = 1))
        return latent_loss

    def _reconstruction_loss(self, f, x_gt):
        """Constructs the reconstruction loss, assuming Gaussian distribution.

        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                _ndims).
            x_gt(tf.Tensor): Ground truth for each example, dimension (None,
                _ndims).
        Returns:
            recon_loss(tf.Tensor): A scalar Tensor for dimension ()
                containing the reconstruction loss.
        """
        recon_loss = None
        ####### Implementation Here ######
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f, x_gt), axis = 1))
        #epsilon = 1e-10
        #recon_loss = tf.reduce_mean(-tf.reduce_sum(
            #x_gt * tf.log(epsilon + f) + (1 - x_gt) * tf.log(epsilon + 1 - f),
            #axis=1))
        return recon_loss

    def loss(self, f, x_gt, z_mean, z_var):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss.

        Args:
            f (tf.Tensor): Decoded image for each example, dimension (None,
                _ndims).
            x_gt (tf.Tensor): Ground truth for each example, dimension (None,
                _ndims)
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, _nlatent)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, _nlatent)

        Returns:
            total_loss: Tensor for dimension (). Sum of
                latent_loss and reconstruction loss.
        """
        total_loss = None
        ####### Implementation Here ######
        latent_loss = self._latent_loss(z_mean, z_var)
        recon_loss = self._reconstruction_loss(f, x_gt)
        total_loss = latent_loss + recon_loss
        return total_loss

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        train_op = None
        ####### Implementation Here ######
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op

    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        out = None
        ####### Implementation Here ######
        out = self.session.run(self.outputs_tensor, feed_dict={self.z: z_np})
        return out

