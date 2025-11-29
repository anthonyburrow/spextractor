import numpy as np
import pytest
from SpectrumCore.Spectrum import Spectrum

from spextractor.math.GaussianProcessModel import GaussianProcessModel
from spextractor.math.InterpolationModel import InterpolationModel
from spextractor.math.SplineModel import SplineModel


@pytest.fixture(scope='module')
def synthetic_spectrum():
    wave = np.linspace(5800.0, 6500.0, 200)
    # Simple absorption feature near 6150
    depth = 0.5
    sigma = 50.0
    flux = 1.0 - depth * np.exp(-((wave - 6150.0) ** 2) / (2.0 * sigma**2))
    error = 0.02 * np.ones_like(flux)
    data = np.c_[wave, flux, error]
    return Spectrum(data)


def test_spline_model_fit_predict(synthetic_spectrum):
    model: InterpolationModel = SplineModel()
    model.fit(synthetic_spectrum)
    pred_wave = np.linspace(5850.0, 6450.0, 50)
    result = model.predict(pred_wave)
    assert isinstance(result, np.ndarray)
    assert result.shape == pred_wave.shape


def test_gp_model_fit_predict(synthetic_spectrum):
    model: InterpolationModel = GaussianProcessModel()
    model.fit(synthetic_spectrum)
    pred_wave = np.linspace(5850.0, 6450.0, 50)
    result = model.predict(pred_wave)
    assert isinstance(result, tuple)
    mean, std = result
    assert mean.shape == pred_wave.shape
    assert std.shape == pred_wave.shape
    assert np.all(std >= 0.0)


def test_gp_model_sample_shape(synthetic_spectrum):
    model = GaussianProcessModel()
    model.fit(synthetic_spectrum)
    wave = synthetic_spectrum.wave
    samples = model.sample(wave, n_samples=25)
    # sample_y returns (N_wave, n_samples)
    assert samples.shape == (wave.size, 25)
