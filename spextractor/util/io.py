import numpy as np
from pandas import isna


def load_spectra(filename):
    if filename[-5:] == '.fits':
        return _load_fits(filename)
    else:
        return _load_other(filename)


def _load_fits(filename):
    from astropy.io import fits
    import astropy.units as u
    import astropy.wcs as fitswcs
    from specutils.spectra import Spectrum1D

    hdu = fits.getheader(filename)
    flux = fits.getdata(filename)

    # Handle multi-array flux from fits by picking that with
    # larger values (presumably smaller is flux error)
    if len(flux.shape) > 1:
        f1, f2 = flux
        if f2[0][0] < f1[0][0]:
            flux = f1[0]
        else:
            flux = f2[0]

    if not ('WAT0_001' in hdu) or hdu['WAT0_001'] == 'system=equispec':
        uflux = u.erg / (u.cm ** 2 * u.s)

        try:
            if 'CDELT1' in hdu:
                cdelt1 = hdu['CDELT1']
            else:
                cdelt1 = hdu['CD1_1']
        except KeyError as e:
            raise e

        if 'CUNIT1' in hdu:
            cunit1 = hdu['CUNIT1']
        else:
            cunit1 = 'Angstrom'

        crval1 = hdu['CRVAL1']
        ctype1 = hdu['CTYPE1']
        crpix1 = hdu['CRPIX1']
        my_wcs = fitswcs.WCS(header={
            'CDELT1': cdelt1,
            'CRVAL1': crval1,
            'CUNIT1': cunit1,
            'CTYPE1': ctype1,
            'CRPIX1': crpix1
        })

        spec = Spectrum1D(flux=flux * uflux, wcs=my_wcs)

        wavel = np.array(spec.wavelength)
        flux = np.array(spec.flux)

    elif hdu['WAT0_001'] == 'system=multispec':
        # This is a terrible .fits system that doesn't contain
        # delta-wavelength info, so I improvised.
        wavel = ''
        for line in (x for x in hdu.keys() if x[:4] == 'WAT2'):
            wavel += hdu[line]
        wavel = wavel.split()
        wavel[-1] = wavel[-1][:-1]   # get rid of end quote
        wavel = wavel[16:]

        # Split error values due to errors in header key-value reading
        wave_to_add = []
        for w in reversed(wavel):
            if w.count('.') == 2:
                w_all = w.split('.')
                # Assumes wavelength is between 1000-9999
                wave_to_add.append('%s.%s' % (w_all[0], w_all[1][:-4]))
                wave_to_add.append('%s.%s' % (w_all[1][-4:], w_all[2]))
                wavel.remove(w)

        # For some reason the wavelength is sometimes reversed
        if float(wavel[0]) > float(wavel[-1]):
            flux = list(reversed(flux))

        wavel += wave_to_add
        wavel = [float(x) for x in wavel]
        wavel = sorted(wavel)

        wavel = np.array(wavel)
        flux = np.array(flux)

    flux_err = np.zeros(len(flux))

    return wavel, flux, flux_err


def _load_other(filename):
    '''Load general data files'''
    read_err = False
    try:
        data = np.genfromtxt(filename)
    except Exception as e:
        read_err = True
        prev = e

    if read_err:
        try:
            from astropy.io import ascii
            data = ascii.read(filename)
        except Exception as e:
            print(prev.message, e.message, filename)
            raise e

    nan_mask = ~np.isnan(data).any(axis=1)
    data = data[nan_mask]

    wavel = data[:, 0]
    flux = data[:, 1]

    try:
        flux_err = data[:, 2]
    except IndexError:
        print('No flux uncertainties found while reading file')
        flux_err = np.zeros(len(flux))

    if np.all(isna(flux_err)):
        flux_err = np.zeros(len(flux_err))

    return wavel, flux, flux_err
