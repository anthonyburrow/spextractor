import numpy as np

from ..physics.doppler import deredshift
from ..physics.deredden import dered_ccm


def preprocess(data, *args, **kwargs):
    data = remove_nan(data)
    data = remove_zeros(data, *args, **kwargs)

    data = deredshift(data, *args, **kwargs)
    data = prune(data, *args, **kwargs)
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


def prune(data, wave_range=None, *args, **kwargs):
    if wave_range is None:
        return data

    mask = (wave_range[0] <= data[:, 0]) & (data[:, 0] <= wave_range[1])
    return data[mask]


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
