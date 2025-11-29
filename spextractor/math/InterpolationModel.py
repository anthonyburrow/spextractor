from abc import ABC, abstractmethod

import numpy as np
from SpectrumCore import Spectrum


class InterpolationModel(ABC):
    """
    Abstract base class for spectrum interpolation models.

    Subclasses must implement fit() and predict() methods.
    """

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
