from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from ops import linear
from ops import gmm_sample
from utils import make_batches
from utils import disp_scatter


class D2GAN(object):
    """Dual Discriminator Generative Adversarial Nets for 2D data
    """

    def __init__(self,
                 model_name="D2GAN",
                 num_z=256,  # number of noise variables
                 hidden_size=128,
                 alpha=1.0,  # coefficient - regularization constant of D1
                 beta=1.0,  # coefficient - regularization constant of D2
                 mix_coeffs=(0.5, 0.5),
                 mean=((0.0, 0.0), (1.0, 1.0)),
                 cov=((0.1, 0.1), (0.1, 0.1)),
                 batch_size=512,
                 learning_rate=0.0002,
                 num_epochs=25000,
                 disp_freq=5000,
                 random_seed=6789,
                 ):
        self.model_name = model_name
        self.num_z = num_z
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.mix_coeffs = mix_coeffs
        self.mean = mean
        self.cov = cov
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.disp_freq = disp_freq
        self.random_seed = random_seed

    def _init(self):
        self.epoch = 0
        self.fig = None
        self.ax = None

        # TensorFlow's initialization
        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.log_device_placement = False
        self.tf_config.allow_soft_placement = True
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

        np.random.seed(self.random_seed)
        with self.tf_graph.as_default():
            tf.set_random_seed(self.random_seed)

    def _build_model(self):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('generator'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
            self.g = self._create_generator(self.z, self.hidden_size)

        self.x = tf.placeholder(tf.float32, shape=[None, 2])

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('discriminator_1') as scope:
            self.d1x = self._create_discriminator(self.x, self.hidden_size)
            scope.reuse_variables()
            self.d1g = self._create_discriminator(self.g, self.hidden_size)
        with tf.variable_scope('discriminator_2') as scope:
            self.d2x = self._create_discriminator(self.x, self.hidden_size)
            scope.reuse_variables()
            self.d2g = self._create_discriminator(self.g, self.hidden_size)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d1_loss = tf.reduce_mean(-self.alpha * tf.log(self.d1x) + self.d1g)
        self.d2_loss = tf.reduce_mean(self.d2x - self.beta * tf.log(self.d2g))
        self.d_loss = self.d1_loss + self.d2_loss
        self.g_loss = tf.reduce_mean(-self.d1g + self.beta * tf.log(self.d2g))

        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='discriminator_1') \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='discriminator_2')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.d_opt = self._create_optimizer(self.d_loss, self.d_params,
                                            self.learning_rate)
        self.g_opt = self._create_optimizer(self.g_loss, self.g_params,
                                            self.learning_rate)

    def _create_generator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'g_hidden1'))
        hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden2'))
        out = linear(hidden, 2, scope='g_out')
        return out

    def _create_discriminator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1', ))
        out = tf.nn.softplus(linear(hidden, 1, scope='d_out'))
        return out

    def _create_optimizer(self, loss, var_list, initial_learning_rate):
        return tf.train.AdamOptimizer(initial_learning_rate,
                                      beta1=0.5).minimize(loss, var_list=var_list)

    def fit(self):
        if (not hasattr(self, 'epoch')) or self.epoch == 0:
            self._init()
            with self.tf_graph.as_default():
                self._build_model()
                self.tf_session.run(tf.global_variables_initializer())

        while self.epoch < self.num_epochs:
            # update discriminator
            x = gmm_sample(self.batch_size, self.mix_coeffs, self.mean, self.cov)
            z = np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
            d1x, d2x, d1_loss, d2_loss, d_loss, _ = self.tf_session.run(
                [self.d1x, self.d2x, self.d1_loss, self.d2_loss, self.d_loss, self.d_opt],
                feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
                           self.z: np.reshape(z, [self.batch_size, self.num_z]),
                           })

            # update generator
            z = np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
            g_loss, _ = self.tf_session.run(
                [self.g_loss, self.g_opt],
                feed_dict={self.z: np.reshape(z, [self.batch_size, self.num_z])})

            print("Epoch: [%4d/%4d] d1_loss: %.8f, d2_loss: %.8f,"
                  " d_loss: %.8f, g_loss: %.8f" % (self.epoch, self.num_epochs,
                                                   d1_loss, d2_loss, d_loss, g_loss))
            self.epoch += 1

            if self.epoch % self.disp_freq == 0:
                self.display(num_samples=1000)

    def generate(self, num_samples=1000):
        zs = np.random.normal(0.0, 1.0, [num_samples, self.num_z])
        g = np.zeros([num_samples, 2])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            g[batch_start:batch_end] = self.tf_session.run(
                self.g,
                feed_dict={
                    self.z: np.reshape(zs[batch_start:batch_end],
                                       [batch_end - batch_start, self.num_z])
                }
            )
        return g

    def display(self, num_samples=1000):
        x = gmm_sample(num_samples, self.mix_coeffs, self.mean, self.cov)
        g = self.generate(num_samples=num_samples)
        self.fig, self.ax = disp_scatter(x, g, fig=self.fig, ax=self.ax)
        self.fig.tight_layout()
        self.fig.savefig("output\{}.png".format(self.epoch))
