import numpy as np

from ..physics.doppler import deredshift
from ..physics.deredden import dered_ccm
from ..physics.telluric import telluric_to_remove
from .interpolate import power_law, generic


def preprocess(data, *args, **kwargs):
    data = remove_nan(data)
    data = remove_zeros(data, *args, **kwargs)

    data = remove_telluric(data, *args, **kwargs)   # TODO: AFTER MANGLING
    data = deredshift(data, *args, **kwargs)
    data = prune(data, *args, **kwargs)
    data = mangle(data, *args, **kwargs)
    data = deredden(data, *args, **kwargs)

    return data


def remove_nan(data):
    """Remove NaN values."""
    nan_mask = ~np.isnan(data).any(axis=1)
    return data[nan_mask]


def remove_zeros(data, remove_zeros=True, *args, **kwargs):
    """Remove zero-flux values."""
    if not remove_zeros:
        return data

    mask = data[:, 1] != 0.
    return data[mask]


def remove_telluric(data, remove_telluric=False, *args, **kwargs):
    """Remove telluric features before redshift correction."""
    if not remove_telluric:
        return data

    wave = data[:, 0]
    flux = data[:, 1]

    for feature in telluric_to_remove:
        min_ind, max_ind = np.searchsorted(wave, feature)
        if min_ind == max_ind:
            # Feature completely outside wavelengths, ignore it
            continue
        if min_ind == 0 or max_ind == len(wave):
            # Spectrum begins/ends inside telluric, 
            # remove instead since no line can be made
            telluric_mask = (feature[0] <= wave) & (wave <= feature[1])
            data = data[~telluric_mask]
            continue

        telluric_inds = np.arange(min_ind, max_ind)

        min_ind -= 1
        slope = (flux[max_ind] - flux[min_ind]) / (wave[max_ind] - wave[min_ind])
        flux[telluric_inds] = slope * (wave[telluric_inds] - wave[min_ind]) + flux[min_ind]

    return data


def prune(data, wave_range=None, *args, **kwargs):
    if wave_range is None:
        return data

    mask = (wave_range[0] <= data[:, 0]) & (data[:, 0] <= wave_range[1])
    return data[mask]


def _convert_phase_to_mjd(time, t_Bmax=None, phot_file=None, *args, **kwargs):
    if t_Bmax is not None:
        return t_Bmax + time

    # Get t_Bmax from Snoopy model using phot_file


def _interpolate_photometry(time, band, mag_table, phot_interp=None,
                            *args, **kwargs):
    mags = mag_table[band]

    mask = mags < 90.
    mags = mags[mask]
    times = mag_table['MJD'][mask]

    if time < times.min() or time > times.max():
        msg = f'Time {time} for {band} band outside observed light-curve range'
        return np.nan

    input_table = np.c_[times, mags]

    if phot_interp is None:
        phot_interp = 'quadratic'

    if phot_interp == 'powerlaw':
        interp_mag = power_law(time, input_table)
    else:
        interp_mag = generic(time, input_table, method=phot_interp)

    return interp_mag


def mangle(data, spex=None, z=None, phot_file=None, time=None, time_format=None,
           *args, **kwargs):
    if phot_file is None:
        return data

    if time is None:
        # Optionally pick the closest time given in phot_file
        print('`time` was not provided, so spectrum could not be mangled.')
        return data

    try:
        import snpy
        from snpy.mangle_spectrum import mangle_spectrum2
        from snpy.filters import fset
    except ImportError:
        print('Snoopy not able to be imported - Spectrum will not be mangled')
        return data

    if z is None:
        z = 0.

    # Ensure time is mjd here
    if time_format == 'phase':
        time = _convert_phase_to_mjd(time, phot_file=phot_file, *args, **kwargs)

    # Use snpy to load photometry and get mags in desired format
    snpy_obj = snpy.get_sn(phot_file)

    bands = list(snpy_obj.restbands)
    eff_waves = [fset[b].eff_wave(data[:, 0], data[:, 1]) for b in bands]

    bands = [b for w, b in zip(eff_waves, bands) if not np.isnan(w)]
    eff_waves = [w for w in eff_waves if not np.isnan(w)]

    bands = [b for _, b in sorted(zip(eff_waves, bands))]

    res = snpy_obj.get_mag_table(bands)

    mags = [_interpolate_photometry(time, b, res, *args, **kwargs)
            for b in bands]
    mags = np.array(mags)

    mask = ~np.isnan(mags)
    bands = [bands[i] for i in range(len(bands)) if mask[i]]
    mags = mags[mask]

    wave = data[:, 0]
    flux = data[:, 1]

    # These are defined as such in snpy.sn.bolometric() (written explicitly here)
    refband = None   # This becomes bands[-1] in Snoopy, sorta
    init = [1. for _ in bands]   # This init should be replaced by something more efficient

    mflux, ave_wave, pars, mangle_bands = \
        mangle_spectrum2(wave, flux, bands, mags,
                         z=z, normfilter=refband, init=init)
    mflux = mflux.squeeze()

    data[:, 1] = mflux
    data[:, 2] *= mflux / flux
    spex.mangle_bands = mangle_bands
    return data


def deredden(data, host_EBV=None, host_RV=None, MW_EBV=None, MW_RV=3.1,
             *args, **kwargs):
    """Correct for extinction."""
    # Milky Way extinction
    if MW_EBV is not None and MW_EBV != 0. and MW_RV is not None:
        data[:, 1] = dered_ccm(data, E_BV=MW_EBV, R_V=MW_RV)

    # Host extinction
    if host_EBV is not None and host_EBV != 0. and host_RV is not None:
        data[:, 1] = dered_ccm(data, E_BV=host_EBV, R_V=host_RV)

    return data
