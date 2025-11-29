c = 299.792458  # 10^3 km/s


def velocity(lam: float, lam_err: float, lam0: float) -> tuple[float, float]:
    """Relativistic Doppler shift (reverse)."""
    l_ratio = lam / lam0
    vel = -c * (l_ratio**2 - 1.0) / (l_ratio**2 + 1.0)
    vel_err = 4.0 * c * l_ratio / (l_ratio**2 + 1.0) ** 2 * lam_err / lam0

    return vel, vel_err
