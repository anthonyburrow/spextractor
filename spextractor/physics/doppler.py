c = 299.792458   # 10^3 km/s


def velocity(lam, lam_err, lam0):
    '''Relativistic Doppler shift (reverse).'''
    l_ratio = lam / lam0
    vel = -c * (l_ratio**2 - 1.) / (l_ratio**2 + 1.)
    vel_err = 4. * c * l_ratio / (l_ratio**2 + 1.)**2 * lam_err / lam0

    return vel, vel_err
