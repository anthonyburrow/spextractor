import numpy as np
from .util.interpolate import interp_linear


def downsample(wavelength, flux, flux_err, binning=1, method='weighted'):
    assert binning >= 1, 'Downsampling factor must be >= 1'

    if binning == 1:
        return wavelength, flux, flux_err

    assert wavelength.shape == flux.shape
    assert wavelength.shape == flux_err.shape

    if method == 'weighted':
        return _downsample_average(wavelength, flux, flux_err, binning)
    elif method == 'remove':
        assert isinstance(binning, int), \
            'Downsampling factor must be integer for removal method'
        return _downsample_remove(wavelength, flux, flux_err, binning)


def _downsample_remove(wavelength, flux, flux_err, binning):
    new_wavelength = wavelength[::binning]
    new_flux = flux[::binning]
    new_flux_err = flux_err[::binning]

    return new_wavelength, new_flux, new_flux_err


def _downsample_average(wave, flux, flux_err, binning):
    n_points = wave.shape[0]
    n_bins = int(np.around(n_points / binning))
    endpoint_wave, bin_size = np.linspace(wave[0], wave[-1], n_bins + 1,
                                          retstep=True)

    new_wavelength = 0.5 * (endpoint_wave[:-1] + endpoint_wave[1:])

    endpoint_flux, endpoint_flux_var = \
        interp_linear(endpoint_wave, wave, flux, flux_err)

    new_flux = []
    new_flux_var = []
    for i in range(n_bins):
        # Get values for individual bin
        bin_mask = (endpoint_wave[i] < wave) & (wave < endpoint_wave[i + 1])

        bin_wave = np.zeros(bin_mask.sum() + 2)
        bin_flux = np.zeros_like(bin_wave)
        bin_flux_var = np.zeros_like(bin_wave)

        bin_wave[[0, -1]] = endpoint_wave[i:i + 2]
        bin_flux[[0, -1]] = endpoint_flux[i:i + 2]
        bin_flux_var[[0, -1]] = endpoint_flux_var[i:i + 2]

        bin_wave[1:-1] = wave[bin_mask]
        bin_flux[1:-1] = flux[bin_mask]
        bin_flux_var[1:-1] = flux_err[bin_mask]**2

        # Get flux/flux error associated with integrated flux
        dlam = bin_wave[1:] - bin_wave[:-1]
        lamF = bin_wave * bin_flux
        lamF_var = bin_wave**2 * bin_flux_var

        flux_terms = dlam * (lamF[:-1] + lamF[1:])
        int_flux = 0.5 * flux_terms.sum()
        new_flux.append(int_flux)

        var_terms = dlam**2 * (lamF_var[:-1] + lamF_var[1:])
        int_var = 0.25 * var_terms.sum()
        new_flux_var.append(int_var)

    new_flux = np.array(new_flux)
    new_flux_var = np.array(new_flux_var)

    lam_dlam = new_wavelength * bin_size
    new_flux /= lam_dlam
    new_flux_err = np.sqrt(new_flux_var) / lam_dlam

    # Filter for flux = np.nan?

    return new_wavelength, new_flux, new_flux_err


def get_downsample_factor(wavelength, R, round_factor=False):
    interval = (wavelength[-1] - wavelength[0]) / (len(wavelength) - 1)
    lam = (wavelength[-1] + wavelength[0]) / 2
    dlam = lam / R

    if round_factor:
        factor = int(np.around(dlam / interval))
    else:
        factor = dlam / interval

    if factor < 1:
        return 1

    return factor
