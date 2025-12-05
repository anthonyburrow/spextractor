from abc import ABC, abstractmethod

import numpy as np
from SpectrumCore import Spectrum


class InterpolationModel(ABC):
    """
    Abstract base class for spectrum interpolation models.

    Subclasses must implement fit() and predict() methods.

    This base class handles X (wavelength) standardization automatically
    to improve numerical conditioning for all interpolation methods.
    """

    def __init__(self):
        """Initialize normalization parameters."""
        self._x_mean: float | None = None
        self._x_std: float | None = None

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize input wavelengths using stored mean and std.

        Parameters
        ----------
        X : np.ndarray
            Input wavelength values.

        Returns
        -------
        np.ndarray
            Standardized wavelength values.
        """
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError('Model must be fit before normalizing X.')
        return (X - self._x_mean) / self._x_std

    def _store_normalization(self, X: np.ndarray) -> np.ndarray:
        """
        Store normalization parameters and return normalized X.

        Parameters
        ----------
        X : np.ndarray
            Training wavelength values.

        Returns
        -------
        np.ndarray
            Standardized wavelength values.
        """
        self._x_mean = float(X.mean())
        self._x_std = float(X.std())
        return self._normalize_x(X)

    @abstractmethod
    def fit(self, spectrum: Spectrum):
        """
        Fit the model to a Spectrum object.

        Parameters
        ----------
        spectrum : SpectrumCore.Spectrum
            Spectrum object containing wavelength and flux data.
        """
        pass

    @abstractmethod
    def predict(
        self, X_pred: np.ndarray
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict flux (and optionally uncertainty) for given wavelengths.

        Parameters
        ----------
        X_pred : np.ndarray
            Wavelength values to predict at.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            Predicted flux values, or (flux, uncertainty) tuple if model
            provides uncertainties.
        """
        pass
