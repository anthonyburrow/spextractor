import numpy as np

from spextractor import Spextractor


def test_initialization(file_optical):
    spex = Spextractor(file_optical)

    wave = spex.spectrum.wave
    flux = spex.spectrum.flux
    error = spex.spectrum.error

    assert wave[0] == 3476.0
    assert flux[0] == 1.3898256e-16 / spex.spectrum._flux_norm
    assert error[0] == 1.1478313e-16 / spex.spectrum._flux_norm


def test_preprocessing(file_optical):
    params = {
        'z': 0.0001,
    }

    spex = Spextractor(file_optical, **params)

    wave = spex.spectrum.wave
    flux = spex.spectrum.flux

    assert wave[0] == 3476.0 / (params['z'] + 1.0)
    assert flux.max() == 1.0


def test_modeling(file_optical):
    spex = Spextractor(file_optical)

    spex.create_model(downsampling=3.0)

    assert spex.model is not None

    wave_pred = np.linspace(5500.0, 6000.0, 100)
    mean, var = spex.predict(wave_pred)


def test_process(file_optical):
    spex = Spextractor(file_optical)

    spex.create_model(downsampling=3.0)

    SiII = 'Si II 6150A'
    spex.process(features=(SiII,))

    assert 1.0 <= spex.vel[SiII] < 30.0
    assert 0.0 <= spex.vel_err[SiII] < spex.vel[SiII]
    assert 0.0 < spex.pew[SiII] < 200.0
    assert 0.0 < spex.pew_err[SiII] < spex.pew[SiII]
