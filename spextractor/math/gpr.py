from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


def model(spectrum, logger=None):
    """Calculate the GPy model for given data.

    Uses GPy to determine a Gaussian process model based on given training
    data and optimized hyperparameters.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum object to model.

    Returns
    -------
    m : GPy.models.GPRegression
        Fitted GPy model.
    kernel : GPy.kern
        Kernel with optimized hyperparameters.

    """
    kernel = (
        ConstantKernel(0.5, (1e-2, 1e2)) *
        Matern(length_scale=300., length_scale_bounds=(50., 1e4), nu=1.5) +
        WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e0))
    )

    # Add flux uncertainty to kernel diagonal
    if spectrum.has_error:
        alpha = spectrum.error**2
    else:
        alpha = 1e-6

    # Create model
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=False, n_restarts_optimizer=0
    )

    logger.info('Created GP model')

    # Optimize model
    logger.info('Optimizing hyperparameters...')

    X = spectrum.wave
    y = spectrum.flux
    gpr.fit(X.reshape(-1, 1), y)

    logger.info(gpr.kernel_)

    return gpr


def predict(X_pred, gpr):
    y_pred, y_std = gpr.predict(X_pred.reshape(-1, 1), return_std=True)

    return y_pred, y_std
