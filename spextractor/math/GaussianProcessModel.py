import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from SpectrumCore import Spectrum

from .InterpolationModel import InterpolationModel


class GaussianProcessModel(InterpolationModel):
    """
    Gaussian Process regression model for spectra using scikit-learn.

    Usage:
        model = GaussianProcessModel(logger)
        model.fit(spectrum)
        y_pred, y_std = model.predict(wavelengths)
    """

    def __init__(self, logger=None):
        """Initialize Gaussian Process model.

        Parameters
        ----------
        logger : logging.Logger | None, optional
            Logger for diagnostic output. If None, logging is suppressed.
        """
        self._logger = logger
        self._model: GaussianProcessRegressor | None = None

        constant_kernel = ConstantKernel(0.5, (1e-2, 1e2))
        matern_kernel = Matern(
            length_scale=300.0,
            length_scale_bounds=(50.0, 1e4),
            nu=1.5,
        )
        white_kernel = WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-5, 1e0),
        )
        self.kernel = constant_kernel * matern_kernel + white_kernel

    def fit(self, spectrum: Spectrum) -> GaussianProcessRegressor:
        """
        Fit the GP model to a Spectrum object.
        """
        if spectrum.has_error:
            alpha = spectrum.error**2
        else:
            alpha = 1e-6

        self._model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=0,
        )

        if self._logger:
            self._logger.info('Created GP model')
            self._logger.info('Optimizing hyperparameters...')

        X = spectrum.wave
        y = spectrum.flux
        self._model.fit(X.reshape(-1, 1), y)
        if self._logger:
            self._logger.info(self._model.kernel_)

        return self._model

    def predict(self, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict flux and uncertainty for given wavelengths.
        """
        if self._model is None:
            raise RuntimeError('Model must be fit before prediction.')

        y_pred, y_std = self._model.predict(  # type: ignore
            X_pred.reshape(-1, 1), return_std=True
        )
        return y_pred, y_std

    def sample(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Draw samples from the GP posterior at input wavelengths.

        Parameters
        ----------
        X : np.ndarray
            Wavelength values (1D array) at which to sample.
        n_samples : int, optional
            Number of posterior samples to draw. Default is 100.

        Returns
        -------
        np.ndarray
            Array of shape (len(X), n_samples) containing sampled flux curves.
        """
        if self._model is None:
            raise RuntimeError('Model must be fit before sampling.')
        return self._model.sample_y(X.reshape(-1, 1), n_samples)
