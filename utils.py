from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


def disp_scatter(x, g, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], s=50, marker='+', color='r', alpha=0.8, label='real data')
    ax.scatter(g[:, 0], g[:, 1], s=50, marker='o', color='b', alpha=0.8, label='generated data')
    ax.legend()
    return fig, ax
