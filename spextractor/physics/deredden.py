import numpy as np


# Original coefficients from CCM89
c1_ccm = [1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999]
c2_ccm = [0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002]

# New coefficents from O'Donnell (1994)
c1_o94 = [1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
c2_o94 = [0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]


def ccm(wave, strict_ccm=False, *args, **kwargs):
    '''Returns the Cardelli, Clayton, and Mathis (CCM) reddening curve.

    Args:
      wave (float array): wavelength in Angstroms
      strict_ccm (bool): If True, return original CCM (1989), othewise
                         apply updates from O'Donnel (1994)

    Returns:
      2-tupe (a,b):
      The coeffients such that :math:`A_lambda/A_V = a + b/R_V`
    '''
    x = 10000. / wave   # Convert to inverse microns
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # Infrared correction
    mask = (0.3 < x) & (x < 1.1)
    a[mask] = 0.574 * x[mask]**1.61
    b[mask] = -0.527 * x[mask]**1.61

    # Optical/NIR correction
    mask = (1.1 <= x) & (x < 3.3)
    if np.any(mask):
        if strict_ccm:
            c1 = np.array(c1_ccm)
            c2 = np.array(c2_ccm)
        else:
            c1 = np.array(c1_o94)
            c2 = np.array(c2_o94)

        n_coeff = len(c1)

        y = x[mask] - 1.82
        y_powers = np.ones((len(y), n_coeff))
        y_powers[:, 1] = y
        for i in range(2, n_coeff):
            y_powers[:, i] = y_powers[:, i - 1] * y

        a[mask] = (c1 * y_powers).sum(axis=1)
        b[mask] = (c2 * y_powers).sum(axis=1)

    # Mid-UV correction
    mask = (3.3 <= x) & (x < 8.)
    if np.any(mask):
        y = x[mask]
        F_a = np.zeros_like(y)
        F_b = np.zeros_like(y)

        mask1 = y > 5.9
        if np.any(mask1):
            y1 = y[mask1] - 5.9
            F_a[mask1] = -0.04473 * y1**2 - 0.009779 * y1**3
            F_b[mask1] = 0.2130 * y1**2 + 0.1207 * y1**3

        a[mask] = F_a + 1.752 - 0.316 * y - 0.104 / ((y - 4.67)**2 + 0.341)
        b[mask] = F_b - 3.090 + 1.825 * y + 1.206 / ((y - 4.67)**2 + 0.263)

    # Far-UV correction
    mask = (8. <= x) & (x <= 11.)
    if np.any(mask):
        c1 = [-1.073, -0.628, 0.137, -0.070]
        c2 = [13.670, 4.257, -0.420, 0.374]

        n_coeff = len(c1)

        y = x[mask] - 8.0
        y_powers = np.ones((len(y), n_coeff))
        y_powers[:, 1] = y
        for i in range(2, n_coeff):
            y_powers[:, i] = y_powers[:, i - 1] * y

        a[mask] = (c1 * y_powers).sum(axis=1)
        b[mask] = (c2 * y_powers).sum(axis=1)

    return a, b


def dered_ccm(data, E_BV, R_V=3.1, *args, **kwargs):
    '''Deredden a spectrum with CCM method.'''
    a, b = ccm(data[:, 0], *args, **kwargs)
    A_lambda = E_BV * (a * R_V + b)
    dered_flux = data[:, 1] * 10.**(0.4 * A_lambda)

    return dered_flux
