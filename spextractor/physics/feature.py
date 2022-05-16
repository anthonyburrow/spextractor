import numpy as np
from scipy.integrate import trapz
from scipy import signal

from . import doppler
from ..math import interpolate


def velocity(feat_data, lam0, model, kernel, n_samples=100):
    wave = feat_data[:, 0]
    flux = feat_data[:, 1]

    min_index = flux.argmin()

    # If clear feature not found
    if min_index == 0 or min_index == len(flux) - 1:
        return np.nan, np.nan

    lam_min = wave[min_index]

    # To estimate the error, sample possible spectra from the posterior
    # and find the minima
    samples = model.posterior_samples_f(wave[:, np.newaxis], n_samples,
                                        kern=kernel.copy())
    samples = samples.squeeze()
    min_sample_indices = samples.argmin(axis=0)

    # Exclude points at either end
    min_sample_indices = min_sample_indices[1:-1]
    if len(min_sample_indices) == 0:
        return np.nan, np.nan

    lam_min_err = np.std(wave[min_sample_indices])

    vel, vel_err = doppler.velocity(lam_min, lam_min_err, lam0)

    return vel, vel_err


def pEW(feat_data):
    wave_range = feat_data[[0, -1], 0]

    continuum = interpolate.linear(feat_data[:, 0], feat_data[[0, -1]])
    frac_flux = 1 - feat_data[:, 1] / continuum
    pEW = trapz(frac_flux, x=feat_data[:, 0])

    pEW_stat_err = np.abs(signal.cwt(feat_data[:, 1], signal.ricker, [1])).mean()
    pEW_cont_err = (wave_range[1] - wave_range[0]) * pEW_stat_err
    pEW_err = pEW_stat_err**2 + pEW_cont_err**2

    return pEW, pEW_err


def depth(feat_data):
    """Calculate line depth for feature

    Args:
        wave (ndarray): Wavelength values of feature.
        flux (ndarray): Flux values of feature.
        flux_err (ndarray): Error in flux values of feature.

    Returns:
        depth (float): Depth of line from pseudo-continuum.
        depth_err (float): Error in depth.

    """
    feat_range = feat_data[[0, -1]]

    min_ind = feat_data[:, 1].argmin()

    if min_ind == 0 or min_ind == len(feat_data) - 1:
        return np.nan

    min_wave = np.asarray([feat_data[min_ind, 0]])
    cont, cont_err = interpolate.linear(min_wave, feat_range)

    # Continuum error is extremely large, so depth_err really means nothing
    depth = cont - feat_data[min_ind, 1]
    depth_err = np.sqrt(feat_data[min_ind, 2]**2 + cont_err**2)

    return depth, depth_err
