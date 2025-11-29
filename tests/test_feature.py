import numpy as np
from SpectrumCore.Spectrum import Spectrum

from spextractor.math.GaussianProcessModel import GaussianProcessModel
from spextractor.math.SplineModel import SplineModel
from spextractor.physics.feature import Feature
from spextractor.physics.lines import FEATURE_RANGES, FEATURE_REST_WAVES


def _synthetic_spectrum():
    wave = np.linspace(5800.0, 6500.0, 220)
    depth = 0.5
    sigma = 50.0
    flux = 1.0 - depth * np.exp(-((wave - 6150.0) ** 2) / (2.0 * sigma**2))
    error = 0.02 * np.ones_like(flux)
    data = np.c_[wave, flux, error]
    return Spectrum(data)


def _make_feature(model_type: str):
    spectrum = _synthetic_spectrum()
    if model_type == 'gpr':
        model = GaussianProcessModel()
    else:
        model = SplineModel()
    model.fit(spectrum)

    name = 'Si II 6150A'
    rest_wave = FEATURE_REST_WAVES[name]
    ranges = FEATURE_RANGES[name]

    feature = Feature(name, rest_wave, spectrum, model)
    feature.update_endpoints(ranges['lo_range'], ranges['hi_range'])
    return feature


def test_feature_velocity_minimum_gpr():
    feature = _make_feature('gpr')
    vel, vel_err, draw = feature.velocity(velocity_method='minimum')
    assert not np.isnan(vel)
    assert vel_err >= 0.0
    assert not np.isnan(draw[0])


def test_feature_velocity_minimum_spline():
    feature = _make_feature('spline')
    vel, vel_err, draw = feature.velocity(velocity_method='minimum')
    assert not np.isnan(vel)
    assert vel_err >= 0.0
    assert not np.isnan(draw[0])


def test_feature_pew_gpr():
    feature = _make_feature('gpr')
    pew, pew_err = feature.pEW()
    assert pew > 0.0
    assert pew_err >= 0.0


def test_feature_pew_spline():
    feature = _make_feature('spline')
    pew, pew_err = feature.pEW()
    assert pew > 0.0
    assert pew_err >= 0.0
