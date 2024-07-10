import numpy as np


def gaussian(X, mu, sigma, A, const):
    '''Unnormalized Gaussian function.'''
    return A * np.exp(-0.5 * ((X - mu) / sigma)**2) + const
