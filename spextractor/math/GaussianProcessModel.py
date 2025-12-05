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
        super().__init__()

        self._logger = logger
        self._model: GaussianProcessRegressor | None = None

        self._length_scale_angstroms = 300.0
        self._length_scale_bounds_angstroms = (50.0, 1e4)

    def fit(self, spectrum: Spectrum) -> GaussianProcessRegressor:
        """
        Fit the GP model to a Spectrum object.
        """
        X_norm = self._store_normalization(spectrum.wave)
        y = spectrum.flux

        if self._x_std is None:
            raise RuntimeError('Normalization parameters not set.')
        length_scale_norm = self._length_scale_angstroms / self._x_std
        length_scale_bounds_norm = (
            self._length_scale_bounds_angstroms[0] / self._x_std,
            self._length_scale_bounds_angstroms[1] / self._x_std,
        )

        constant_kernel = ConstantKernel(0.5, (1e-2, 1e2))
        matern_kernel = Matern(
            length_scale=length_scale_norm,
            length_scale_bounds=length_scale_bounds_norm,
            nu=1.5,
        )
        white_kernel = WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-6, 1e0),
        )
        kernel = constant_kernel * matern_kernel + white_kernel

        if spectrum.has_error:
            alpha = spectrum.error**2
        else:
            alpha = 1e-6

        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=0,
        )

        if self._logger:
            self._logger.info('Created GP model')
            self._logger.info('Optimizing hyperparameters...')

        self._model.fit(X_norm.reshape(-1, 1), y)
        if self._logger:
            self._logger.info(self._model.kernel_)

        return self._model

    def predict(self, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict flux and uncertainty for given wavelengths.
        """
        if self._model is None:
            raise RuntimeError('Model must be fit before prediction.')

        X_norm = self._normalize_x(X_pred)
        y_pred, y_std = self._model.predict(  # type: ignore
            X_norm.reshape(-1, 1), return_std=True
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

        X_norm = self._normalize_x(X)
        return self._model.sample_y(X_norm.reshape(-1, 1), n_samples)
