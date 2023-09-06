import numpy as np
from scipy.interpolate import interp1d


def power_law(X: float | np.ndarray, data: np.ndarray) -> float | np.ndarray:
    x_data = data[:, 0]
    y_data = data[:, 1]

    return_float = isinstance(X, float)
    if return_float:
        X = np.array([X])

    Y = np.zeros_like(X)

    ind_X = np.searchsorted(x_data, X) - 1

    # Check if X lie outside x_data
    N = len(x_data)

    Y[ind_X == -1] = y_data[0]
    Y[ind_X == N - 1] = y_data[-1]
    interp_mask = (ind_X != -1) & (ind_X != N - 1)
    ind_X = ind_X[interp_mask]
    ind_X_next = ind_X + 1

    N_interp = len(ind_X)
    if N_interp == 0:
        return Y[0] if return_float else Y

    Y_interp = np.zeros(N_interp)

    # Look for a sign change
    do_log = x_data[ind_X] * x_data[ind_X_next] > 0.

    # Power law interpolation
    y_rat = y_data[ind_X_next][do_log] / y_data[ind_X][do_log]
    x_rat = x_data[ind_X_next][do_log] / x_data[ind_X][do_log]
    power = np.log(y_rat) / np.log(x_rat)

    x_rat_x = X[interp_mask][do_log] / x_data[ind_X][do_log]

    Y_interp[do_log] = y_data[ind_X][do_log] * x_rat_x**power

    if np.all(do_log):
        Y[interp_mask] = Y_interp
        return Y[0] if return_float else Y

    # Linear interpolate across sign changes
    slope = (y_data[ind_X_next][~do_log] - y_data[ind_X][~do_log]) / \
            (x_data[ind_X_next][~do_log] - x_data[ind_X][~do_log] + 1.e-70)

    Y_interp[~do_log] = \
        y_data[ind_X][~do_log] + \
        slope * (X[interp_mask][~do_log] - x_data[ind_X][~do_log])

    Y[interp_mask] = Y_interp

    return Y[0] if isinstance(X, float) else Y


def generic(X: float | np.ndarray, data: np.ndarray, method: str) -> float | np.ndarray:
    interp = interp1d(x=data[:, 0], y=data[:, 1], kind=method)
    return interp(X)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = 20
    sigma = 0.1

    x_data = np.linspace(-10., 10., N)
    y_data = x_data**3. + np.random.normal(scale=sigma, size=N)
    data = np.c_[x_data, y_data]

    x_interp = np.linspace(-10, 10, 15)
    y_interp = generic(x_interp, data)

    fig, ax = plt.subplots()

    ax.plot(x_data, y_data, 'ko')
    ax.plot(x_interp, y_interp, 'ro')

    plt.show()
