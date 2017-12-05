from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import argparse
import numpy as np
import tensorflow as tf

from models import D2GAN

FLAGS = None


def main(_):
    num_mixtures = 8
    radius = 2.0
    std = 0.02
    thetas = np.linspace(0, 2 * np.pi, num_mixtures + 1)[:num_mixtures]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)

    model = D2GAN(
        num_z=FLAGS.num_z,
        hidden_size=FLAGS.hidden_size,
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        mix_coeffs=tuple([1 / num_mixtures] * num_mixtures),
        mean=tuple(zip(xs, ys)),
        cov=tuple([(std, std)] * num_mixtures),
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        num_epochs=FLAGS.num_epochs,
        disp_freq=FLAGS.disp_freq,
        random_seed=6789)
    model.fit()


if __name__ == '__main__':
    # python main.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_z', type=int, default=256,
                        help='Number of latent units.')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Number of hidden units at each layer.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Regularization constant \alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Regularization constant \beta.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Minibatch size.')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=25000,
                        help='Number of epochs.')
    parser.add_argument('--disp_freq', type=int, default=5000,
                        help='Scatter display frequency.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
