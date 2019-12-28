import numpy as np


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


def _downsample_average(wavelength, flux, flux_err, binning):
    # Determine endpoints to bins
    n_points = wavelength.shape[0]
    new_n_points = int(np.around(n_points / binning))
    endpoints = np.linspace(wavelength[0], wavelength[-1], new_n_points + 1)
    n_bins = new_n_points

    # Generate new wavelength array
    new_wavelength = []
    for i in range(n_bins):
        avg_wavelength = 0.5 * (endpoints[i] + endpoints[i + 1])
        new_wavelength.append(avg_wavelength)

    # Generate new flux array
    new_flux = []
    new_flux_err = []

    f_slope = 0
    last_point = None
    next_point = None
    for i in range(n_bins):
        # Get points between endpoints
        to_bin = []
        for j in range(wavelength.shape[0]):
            if endpoints[i] <= wavelength[j] < endpoints[i + 1]:
                to_bin.append((wavelength[j], flux[j], flux_err[j]))
                continue
            if not to_bin:
                continue
            next_point = (wavelength[j], flux[j], flux_err[j])
            break

        if not to_bin:
            new_flux.append(np.nan)
            new_flux_err.append(np.nan)
            continue

        F_integral = 0
        dF_integral = 0

        # Endpoint to first point of bin
        if not i == 0:
            dlam_total = to_bin[0][0] - last_point[0]
            f_slope = (to_bin[0][1] - last_point[1]) / dlam_total

            lam1 = endpoints[i]
            lam2 = to_bin[0][0]
            dlam = lam2 - lam1
            f2 = to_bin[0][1]
            f1 = f2 - f_slope * dlam

            fe2 = to_bin[0][2]
            fe0 = last_point[2]
            lam_ratio = dlam / dlam_total
            fe1_sq = (fe2 * (1 - lam_ratio))**2 + (fe0 * lam_ratio)**2

            F_integral += (f1 * lam1 + f2 * lam2) * dlam
            dF_integral += dlam**2 * (lam1**2 * fe1_sq + lam2**2 * fe2**2)

        # First point to last point
        for j in range(len(to_bin) - 1):
            lam0, f0, fe0 = to_bin[j]
            lam1, f1, fe1 = to_bin[j + 1]
            dlam = lam1 - lam0

            F_integral += (f0 * lam0 + f1 * lam1) * dlam
            dF_integral += dlam**2 * (lam0**2 * fe0**2 + lam1**2 * fe1**2)

        # Last point of bin to endpoint
        if not i == n_bins - 1:
            last_point = to_bin[-1]

            dlam_total = next_point[0] - last_point[0]
            f_slope = (next_point[1] - last_point[1]) / dlam_total

            lam0 = last_point[0]
            lam1 = endpoints[i + 1]
            dlam = lam1 - lam0
            f0 = last_point[1]
            f1 = f0 + f_slope * dlam

            fe0 = last_point[2]
            fe2 = next_point[2]
            lam_ratio = dlam / dlam_total
            fe1_sq = (fe0 * (1 - lam_ratio))**2 + (fe2 * lam_ratio)**2

            F_integral += (f0 * lam0 + f1 * lam1) * dlam
            dF_integral += dlam**2 * (lam0**2 * fe0**2 + lam1**2 * fe1_sq)

        # Solve for flux and flux error values
        lam_dlam = new_wavelength[i] * (endpoints[i + 1] - endpoints[i])

        bin_flux = 0.5 * F_integral / lam_dlam
        new_flux.append(bin_flux)

        bin_flux_err = 0.5 * np.sqrt(dF_integral) / lam_dlam
        new_flux_err.append(bin_flux_err)

    new_wavelength_flux = [(x, y, ye) for x, y, ye in
                           zip(new_wavelength, new_flux, new_flux_err)
                           if y is not np.nan]

    new_wavelength = np.array([x[0] for x in new_wavelength_flux])
    new_flux = np.array([x[1] for x in new_wavelength_flux])
    new_flux_err = np.array([x[2] for x in new_wavelength_flux])

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
