import numpy as np
from scipy.integrate import trapezoid
from scipy import signal
from scipy.optimize import curve_fit

import sys

from . import doppler
from ..math import interpolate
from ..math.functions import gaussian


def velocity(feat_data, rest_wave, spex, velocity_method=None,
             *args, **kwargs):
    if velocity_method is None:
        velocity_method = 'minimum'

    if velocity_method == 'minimum':
        return _velocity_minimum(feat_data, rest_wave, spex, *args, **kwargs)
    if velocity_method == 'blue_edge':
        return _velocity_blue_edge(feat_data, rest_wave, spex, *args, **kwargs)
    else:
        sys.exit('Invalid velocity method given')


def _velocity_minimum(feat_data, rest_wave, spex, n_samples=100,
                      *args, **kwargs):
    wave = feat_data[:, 0]
    flux = feat_data[:, 1]

    min_index = flux.argmin()

    # If clear feature not found
    if min_index == 0 or min_index == len(flux) - 1:
        return np.nan, np.nan

    lam_min = wave[min_index]

    # To estimate the error, sample possible spectra from the posterior
    # and find the minima
    samples = spex.model.posterior_samples_f(wave[:, np.newaxis], n_samples,
                                             kern=spex.kernel.copy())
    samples = samples.squeeze()
    min_sample_indices = samples.argmin(axis=0)

    # Exclude points at either end
    min_sample_indices = min_sample_indices[1:-1]
    if len(min_sample_indices) == 0:
        return np.nan, np.nan

    lam_min_err = np.std(wave[min_sample_indices])

    vel, vel_err = doppler.velocity(lam_min, lam_min_err, rest_wave)

    draw_point = lam_min, flux.min()

    return vel, vel_err, draw_point


def _velocity_blue_edge(feat_data, rest_wave, spex, n_samples=100,
                        feat_profile=None, profile_params=None,
                        *args, **kwargs):
    wave = feat_data[:, 0]
    flux = feat_data[:, 1]

    if feat_profile is None:
        feat_profile = gaussian

    if profile_params is None:
        mu = wave[flux.argmin()]
        sigma = 0.5 * (wave[-1] - wave[0])
        profile_params = (mu, sigma, -0.1, 0.5)
        bounds = (
            (wave[0], 0., -2., 0.),
            (wave[-1], 5. * sigma, 0., 1.)
        )

    params, _ = curve_fit(feat_profile, wave, flux,
                          p0=profile_params, bounds=bounds)
    mu, sigma = params[:2]
    lam = mu - 3. * sigma

    # Calculate lambda error through sampling
    samples = spex.model.posterior_samples_f(wave[:, np.newaxis], n_samples,
                                             kern=spex.kernel.copy())
    samples = samples.squeeze().T

    lam_samples = []
    for sample in samples:
        params_err, _ = curve_fit(feat_profile, wave, sample,
                                  p0=profile_params, bounds=bounds)
        mu_err, sigma_err = params_err[:2]
        lam_samples.append(mu_err - 3. * sigma_err)

    lam_err = np.std(lam_samples)

    # Calculate velocity
    vel, vel_err = doppler.velocity(lam, lam_err, rest_wave)

    if spex._plot:
        spex._ax.plot(wave, feat_profile(wave, *params), 'b-')

    draw_point = lam, feat_profile(lam, *params)

    return vel, vel_err, draw_point


def pEW(feat_data):
    wave_range = feat_data[[0, -1], 0]

    continuum, _ = interpolate.linear(feat_data[:, 0], feat_data[[0, -1]])
    frac_flux = 1 - feat_data[:, 1] / continuum
    pEW = trapezoid(frac_flux, x=feat_data[:, 0])

    pEW_stat_err = np.abs(signal.cwt(feat_data[:, 1], signal.ricker, [1])).mean()
    pEW_cont_err = (wave_range[1] - wave_range[0]) * pEW_stat_err
    pEW_err = np.sqrt(pEW_stat_err**2 + pEW_cont_err**2)

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
