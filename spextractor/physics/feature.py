import numpy as np
from scipy.integrate import trapz
from scipy import signal
from sklearn.cluster import DBSCAN, MeanShift

from . import doppler
from ..math import interpolate


def velocity(wave, flux, lam0, model, kernel, n_samples=100):
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


def velocity_high(wave, flux, lam0, model, kernel, method='MeanShift',
                  n_samples=100):
    method = method.lower()
    methods = ('dbscan', 'meanshift')
    assert method in ('dbscan', 'meanshift'), \
        f'Invalid method "{method}"; valid are {methods}'

    min_index = flux.argmin()

    # If feature not clearly found
    if min_index == 0 or min_index == flux.shape[0] - 1:
        return [], [], np.nan, np.nan, [], []

    # To estimate the error, sample possible spectra from the posterior
    # and find the minima.
    samples = model.posterior_samples_f(wave[:, np.newaxis], n_samples,
                                        kern=kernel.copy())
    samples = samples.squeeze()

    minima_samples = []
    for i in range(samples.shape[1]):
        positions = signal.argrelmin(samples[:, i], order=10)[0]
        minima_samples.extend(positions)

    minima_samples = np.array(minima_samples)[:, np.newaxis]

    if method == 'dbscan':
        labels = DBSCAN(eps=1).fit_predict(minima_samples)
    elif method == 'meanshift':
        labels = MeanShift(10).fit_predict(minima_samples)

    lambdas = []
    lambdas_err = []
    vel_hv = []
    vel_hv_err = []

    for x in np.unique(labels):
        if x == -1:
            # This is the "noise" label in DBSCAN
            continue

        matching = labels == x
        if matching.sum() < 5:
            # This is just noise
            continue

        min_index = minima_samples[matching]
        lam_min = np.mean(wave[min_index])
        lam_min_err = np.std(wave[min_index])

        lambdas.append(lam_min)
        lambdas_err.append(lam_min_err)

        vel, vel_err = doppler.velocity(lam_min, lam_min_err, lam0)
        vel_hv.append(vel)
        vel_hv_err.append(vel_err)

    return lambdas, lambdas_err, vel_hv, vel_hv_err


def pEW(wave, flux):
    wave_range = wave[[0, -1]]
    flux_range = flux[[0, -1]]

    continuum = interpolate.linear(wave, wave_range, flux_range)
    frac_flux = 1 - flux / continuum
    pEW = trapz(frac_flux, x=wave)

    pEW_stat_err = np.abs(signal.cwt(flux, signal.ricker, [1])).mean()
    pEW_cont_err = (wave_range[1] - wave_range[0]) * pEW_stat_err
    pEW_err = pEW_stat_err**2 + pEW_cont_err**2

    return pEW, pEW_err


def depth(wave, flux, flux_err=None):
    """Calculate line depth for feature

    Args:
        wave (ndarray): Wavelength values of feature.
        flux (ndarray): Flux values of feature.
        flux_err (ndarray): Error in flux values of feature.

    Returns:
        depth (float): Depth of line from pseudo-continuum.
        depth_err (float): Error in depth.

    """
    wave_range = wave[[0, -1]]
    flux_range = flux[[0, -1]]
    flux_err_range = flux_err[[0, -1]]

    min_ind = flux.argmin()

    if min_ind == 0 or min_ind == flux.shape[0] - 1:
        return np.nan

    cont, cont_err = interpolate.linear(wave[min_ind],
                                        wave_range, flux_range, flux_err_range)

    # Continuum error is extremely large, so depth_err really means nothing
    depth = cont - flux[min_ind]
    depth_err = np.sqrt(flux_err[min_ind]**2 + cont_err**2)

    return depth, depth_err
