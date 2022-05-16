import numpy as np


def linear(x_new, data):
    '''

    '''
    x = data[:, 0]
    y = data[:, 1]
    y_err = data[:, 2]

    are_errors = np.any(y_err)

    lower_mask = x[0] < x_new
    upper_mask = x_new < x[-1]
    total_mask = lower_mask * upper_mask
    x_check = x_new[total_mask]

    y_new = np.zeros_like(x_new)
    y_new[~lower_mask] = y[0]
    y_new[~upper_mask] = y[-1]

    ind_upper = np.searchsorted(x, x_check)
    ind_lower = ind_upper - 1

    dx_total = x[ind_upper] - x[ind_lower]
    slope = (y[ind_upper] - y[ind_lower]) / dx_total

    dx = x_check - x[ind_lower]
    y_new[total_mask] = y[ind_lower] + dx * slope

    if not are_errors:
        return y_new, y_err

    y_var = y_err**2
    y_var_new = np.zeros_like(x_new)
    y_var_new[~lower_mask] = y_var[0]
    y_var_new[~upper_mask] = y_var[-1]

    upper_dx = x[ind_upper] - x_check
    upper_err_mask = upper_dx < dx
    lower_err_mask = ~upper_err_mask

    slope_var = (y_var[ind_upper] + y_var[ind_lower]) / dx_total**2

    # some absolute BS to do nested masking
    # - In hindsight why couldn't I multiply the masks together?
    lower_err_ind = tuple([a[lower_err_mask] for a in np.where(total_mask)])
    upper_err_ind = tuple([a[upper_err_mask] for a in np.where(total_mask)])

    y_var_new[lower_err_ind] = \
        y_var[ind_lower][lower_err_mask] + \
        dx[lower_err_mask]**2 * slope_var[lower_err_mask]
    y_var_new[upper_err_ind] = \
        y_var[ind_upper][upper_err_mask] + \
        upper_dx[upper_err_mask]**2 * slope_var[upper_err_mask]

    return y_new, y_var_new
