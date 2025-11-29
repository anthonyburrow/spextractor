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

    def __init__(self, logger, k: int = 3):
        """
        Initialize the spline model.

        Parameters
        ----------
        logger : Logger
            Logger object for logged output.
        k : int, optional
            Degree of the spline (default: 3 for cubic spline).
        s : float or None, optional
            Smoothing factor. If None, s = len(data points).
        """
        self._logger = logger
        self._model: UnivariateSpline | None = None
        self.k = k

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
        X = spectrum.wave
        y = spectrum.flux

        # Use error as weights if available
        if spectrum.has_error:
            sigma = spectrum.error.mean()
            w = 1.0 / spectrum.error
        else:
            sigma = 0.05 * y.mean()
            w = None

        s = len(X) * sigma**2
        self._model = UnivariateSpline(X, y, w=w, k=self.k, s=s)

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

        result = self._model(X_pred)
        return np.asarray(result)
