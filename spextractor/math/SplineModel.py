import numpy as np
from scipy.interpolate import UnivariateSpline
from SpectrumCore import Spectrum

from .InterpolationModel import InterpolationModel


class SplineModel(InterpolationModel):
    """
    Univariate spline interpolation model for spectra using scipy.

    Usage:
        model = SplineModel(logger, k=3, s=None)
        model.fit(spectrum)
        y_pred = model.predict(wavelengths)
    """

    def __init__(self, logger=None):
        """Initialize the spline model.

        Parameters
        ----------
        logger : logging.Logger | None, optional
            Logger for diagnostic output. If None, logging suppressed.
        k : int, optional
            Degree of the spline (default: 3 for cubic spline).
        """
        super().__init__()

        self._logger = logger
        self._model: UnivariateSpline | None = None
        self.k = 3

    def fit(self, spectrum: Spectrum) -> UnivariateSpline:
        """
        Fit the spline model to a Spectrum object.

        Parameters
        ----------
        spectrum : SpectrumCore.Spectrum
            Spectrum object to model.

        Returns
        -------
        UnivariateSpline
            Fitted spline model.
        """
        X_norm = self._store_normalization(spectrum.wave)
        y = spectrum.flux

        if spectrum.error is not None:
            sigma = spectrum.error.mean()
            w = 1.0 / spectrum.error
        else:
            sigma = 0.05 * y.mean()
            w = None

        s = len(X_norm) * sigma**2 * 1e3
        self._model = UnivariateSpline(X_norm, y, w=w, k=self.k, s=s)

        if self._logger:
            self._logger.info('Created spline model')
            self._logger.info(f'Spline degree: {self.k}, smoothing: {s}')

        return self._model

    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        """
        Predict flux for given wavelengths.

        Parameters
        ----------
        X_pred : np.ndarray
            Wavelength values to predict at.

        Returns
        -------
        np.ndarray
            Predicted flux values.
        """
        if self._model is None:
            raise RuntimeError('Model must be fit before prediction.')

        X_norm = self._normalize_x(X_pred)
        result = self._model(X_norm)
        return np.asarray(result)
