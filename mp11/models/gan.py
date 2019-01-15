"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.D_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_placeholder)\
                                   .minimize(self.d_loss)
        self.G_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_placeholder)\
                                   .minimize(self.g_loss)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            y = tf.contrib.layers.fully_connected(x, 1024, scope='dis_fc1', activation_fn = tf.nn.elu)
            y = tf.contrib.layers.fully_connected(y, 512, scope='dic_fc2', activation_fn = tf.nn.elu)
            y = tf.contrib.layers.fully_connected(y, 256, scope='dic_fc3', activation_fn=tf.nn.elu)
            y = tf.contrib.layers.fully_connected(y, 1, scope='dic_fc4', activation_fn = tf.nn.sigmoid)
            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # Add the minus since we want to maximize the loss, while the optimizer could only do minimization
        eps = 1e-2
        l = -tf.reduce_mean(tf.log(y + eps) - tf.log(1 - y_hat + eps))
        return l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            x_hat = tf.contrib.layers.fully_connected(z, 256, scope='gen_fc1', activation_fn = tf.nn.elu)
            x_hat = tf.contrib.layers.fully_connected(x_hat, 512, scope='gen_fc2', activation_fn=tf.nn.elu)
            x_hat = tf.contrib.layers.fully_connected(x_hat, 1024, scope='gen_fc3', activation_fn=tf.nn.elu)
            x_hat = tf.contrib.layers.fully_connected(x_hat, self._ndims, scope='gen_fc4', activation_fn=None)
            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        # Actually we need to minimize tf.reduce_mean(tf.log(1-y_hat)) here
        # But as suggested by paper, we had better maximize tf.reduce_mean(tf.log(y_hat)) here
        eps = 1e-2
        l = -tf.reduce_mean(tf.log(y_hat + eps))
        return l

    def generate_samples(self, z):
        x_hat = self.session.run(self.x_hat, feed_dict={self.z_placeholder: z})
        return x_hat
