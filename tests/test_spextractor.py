import numpy as np

from spextractor import Spextractor


def test_initialization(sample_file):
    spex = Spextractor(sample_file)

    wave = spex.spectrum.wave
    flux = spex.spectrum.flux
    error = spex.spectrum.error

    assert wave[0] == 3476.
    assert flux[0] == 1.3898256e-16 / spex.spectrum._flux_norm
    assert error[0] == 1.1478313e-16 / spex.spectrum._flux_norm


def test_preprocessing(sample_file):
    params = {
        'z': 0.0001,
    }

    spex = Spextractor(sample_file, **params)

    wave = spex.spectrum.wave
    flux = spex.spectrum.flux

    assert wave[0] == 3476. / (params['z'] + 1.)
    assert flux.max() == 1.


def test_modeling(sample_file):
    spex = Spextractor(sample_file)

    spex.create_model(downsampling=3.)

    assert spex.model is not None

    wave_pred = np.linspace(5500., 6000., 100)
    mean, var = spex.predict(wave_pred)


def test_process(sample_file):
    spex = Spextractor(sample_file)

    spex.create_model(downsampling=3.)

    SiII = 'Si II 6150A'
    spex.process(features=(SiII,))

    assert 0. <= spex.vel[SiII] < 30.
    assert 0. <= spex.vel_err[SiII] < spex.vel[SiII]
    assert 0. < spex.pew[SiII] < 200.
    assert 0. < spex.pew_err[SiII] < spex.pew[SiII]
