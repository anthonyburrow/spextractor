import numpy as np
from ..math import interpolate


def downsample(data, binning=1., method='weighted'):
    if binning <= 1:
        return data

    if method == 'weighted':
        return _downsample_average(data, binning)
    elif method == 'remove':
        return _downsample_remove(data, binning)


def _downsample_remove(data, binning):
    return data[::binning]


def _downsample_average(data, binning):
    # wave = data[:, 0]
    # flux = data[:, 1]
    # flux_err = data[:, 2]
    n_bins = int(np.around(len(data) / binning))

    endpoint_data = np.zeros((n_bins + 1, 3))
    wave_0, wave_1 = data[0, 0], data[-1, 0]
    endpoint_data[:, 0], bin_size = np.linspace(wave_0, wave_1, n_bins + 1,
                                                retstep=True)

    new_data = np.zeros((n_bins, 3))
    new_data[:, 0] = 0.5 * (endpoint_data[:-1, 0] + endpoint_data[1:, 0])

    endpoint_data[:, 1], endpoint_data[:, 2] = \
        interpolate.linear(endpoint_data[:, 0], data)

    for i in range(n_bins):
        # Get values for individual bin
        bin_mask = (endpoint_data[i, 0] < data[:, 0]) & \
                   (data[:, 0] < endpoint_data[i + 1, 0])
        bin_data = np.zeros((bin_mask.sum() + 2, 3))

        bin_data[[0, -1]] = endpoint_data[i:i + 2]
        bin_data[1:-1] = data[bin_mask]
        bin_data[:, 2] **= 2.

        # Get flux/flux error associated with integrated flux
        bin_wave = bin_data[:, 0]
        bin_flux = bin_data[:, 1]
        bin_flux_var = bin_data[:, 2]

        dlam = bin_wave[1:] - bin_wave[:-1]
        lamF = bin_wave * bin_flux
        lamF_var = bin_wave**2 * bin_flux_var

        flux_terms = dlam * (lamF[:-1] + lamF[1:])
        new_data[i, 1] = 0.5 * flux_terms.sum()

        var_terms = dlam**2 * (lamF_var[:-1] + lamF_var[1:])
        new_data[i, 2] = 0.25 * var_terms.sum()

    lam_dlam = new_data[:, 0] * bin_size
    new_data[:, 1] /= lam_dlam
    new_data[:, 2] = np.sqrt(new_data[:, 2]) / lam_dlam

    return new_data
