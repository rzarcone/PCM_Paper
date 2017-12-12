"""Utilities for memristor functions."""
try:
    from itertools import izip
except: 
    izip = zip
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

PI = np.array(np.pi, dtype='float32')

# Encoding network

def build_multilayer_network(x, n_, f='tanh'):
    """
    Parameters
    ----------
    x : tf.tensor, shape (batch_size, d_in)
        Input Image
    n_ : list of int
        List of number of units in each layer
    f : str
        String describing the Nonlinearity

    Returns
    -------
    v : tf.tensor, shape (batch_size, n_m)
        Encoded version of the image.
    """
    if f == 'tanh':
        f = tf.nn.tanh
    elif f == 'relu':
        f = tf.nn.relu
    # FIXME Pass parameters
    n_layers = len(n_) - 1
    W_ = [tf.get_variable(name='W{}'.format(i),
                          shape=(n_[i], n_[i+1])) for i in range(n_layers)]
    b_ = [tf.get_variable(name='b{}'.format(i),
                          shape=(n_[i+1],)) for i in range(n_layers)]
    in_ = x
    for W, b in zip(W_, b_):
        in_ = f(tf.matmul(in_, W) + b)
    return in_



def gauss_interp(samp, xs, ys, interp_width, ratio=0.75):
    """
    Parameters
    ----------
    samp : tf.tensor (batch_size, n_m)

    xs : tf.tensor (n_p, n_m)
        Grid inputs
    ys : tf.tensor (n_p, n_m)
        Grid outputs
    interp_width : float
        Spacing between the xs

    Returns
    -------
    interp_func : tf.tensor (batch_size, n_m)
    """
    samp = tf.expand_dims(samp, 1)  # (batch, 1, n_m)
    xs = tf.expand_dims(xs, 0)  # (1, n_p, n_m)
    ys = tf.expand_dims(ys, 0)  # (1, n_p, n_m)
    sig = ratio * interp_width  # spacing of xs
    norm_factor = np.sqrt(2 * np.pi) * sig / interp_width
    norm_factor = np.array(norm_factor, dtype='float32')
    return tf.reduce_sum(ys * tf.exp( -0.5 * (samp - xs) ** 2 / sig ** 2) /
                         norm_factor,
                  reduction_indices=1)

def memristor_output(v, eps, vs, mus, sigs, interp_width):
    """
    Parameters
    ----------
    mu, sig, eps : tf.tensor (batch_size, n_m)
        mean, standard deviation, noise

    """
    mean = gauss_interp(v, vs, mus, interp_width)
    sdev = gauss_interp(v, vs, sigs, interp_width)
    return mean + eps * sdev



# Data Iteration Utils

def batch_generator(data, batch_size):
    """
    data : array, shape (n_samples, ...)
        All of your data in a matrix.
    batch_size : int
        Batch size.

    Yields
    ------
    datum : shape (batch_size, ...)
        A batch of data.
    """
    n_samples = data.shape[0]

    num_batches = n_samples / batch_size

    for i in range(num_batches):
        yield data[i * batch_size: (i+1) * batch_size]


def file_batch_generator(files, batch_size, directory, max_batches=100):
    """
    Generator that takes file names and yields batches of images.

    Parameters
    ----------
    files : list of str
        File names
    batch_size : int
        Number of files per batch
    directory : str
        Base directory of the images.
    max_batches : int
        Max number of batches.

    Yields
    -------
    batch : array, shape (batch_size, n_features)
        A batch of images.
    """
    n_samples = len(files)
    num_batches = n_samples / batch_size

    for i in range(num_batches):
        if i >= max_batches:
            break
        file_batch = files[(i + 0) * batch_size:
                           (i + 1) * batch_size]
        batch = None
        for j, fn in enumerate(file_batch):
            img = plt.imread(os.path.join(directory, fn))
            if batch is None:
                n_features = img.size
                batch = np.zeros((batch_size, n_features))
            batch[j] = img.ravel()
        yield batch

def random_generator(n_features, batch_size):
    while True:
        yield np.random.randn(batch_size, n_features)

class FileAndNoiseGenerator(object):
    """
    Class that handles creation of file and noise generator.
    """
    def __init__(self, file_list, base_directory, noise_dim,
                 max_batches=100):
        self.file_list = file_list
        self.base_directory = base_directory
        self.noise_dim = noise_dim
        self.max_batches = max_batches

    def get_generator(self, batch_size=20):
        image_gen = file_batch_generator(
            self.file_list, batch_size, self.base_directory,
            max_batches=self.max_batches)
        rand_gen = random_generator(self.noise_dim, batch_size)
        return izip(image_gen, rand_gen)

class DataAndNoiseGenerator(object):
    """
    Object that Handles Creation of Data Generators.
    """
    def __init__(self, data, noise_dim):
        self.data = data
        self.noise_dim = noise_dim

    def get_generator(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data.shape[0]
        image_gen = batch_generator(self.data, batch_size)
        rand_gen = random_generator(self.noise_dim, batch_size)
        return izip(image_gen, rand_gen)



