from logging import Logger

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from SpectrumCore import Spectrum


def model(spectrum: Spectrum, logger: Logger) -> GaussianProcessRegressor:
    """Calculate the GPy model for given data.

    Uses scikit-learn to determine a Gaussian process regression model based on
    given training data and optimized hyperparameters.

    Parameters
    ----------
    spectrum : SpectrumCore.Spectrum
        Spectrum object to model.
    logger : Logger
        Logger object for logged output.

    Returns
    -------
    m : sklearn.gaussian_process.GaussianProcessRegressor
        Fitted GPy model.
    """
    kernel = ConstantKernel(0.5, (1e-2, 1e2)) * Matern(
        length_scale=300.0, length_scale_bounds=(50.0, 1e4), nu=1.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 1e0))

    # Add flux uncertainty to kernel diagonal
    if spectrum.has_error:
        alpha = spectrum.error**2
    else:
        alpha = 1e-6

    # Create model
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=False, n_restarts_optimizer=0
    )

    logger.info("Created GP model")

    # Optimize model
    logger.info("Optimizing hyperparameters...")

    X = spectrum.wave
    y = spectrum.flux
    gpr.fit(X.reshape(-1, 1), y)

    logger.info(gpr.kernel_)

    return gpr


def predict(
    X_pred: np.ndarray, gpr: GaussianProcessRegressor
) -> tuple[np.ndarray, np.ndarray]:
    y_pred, y_std = gpr.predict(X_pred.reshape(-1, 1), return_std=True)

    return y_pred, y_std
