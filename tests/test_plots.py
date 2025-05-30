import numpy as np

from spextractor import Spextractor
from SpectrumCore.plot import basic_spectrum


def test_prediction(file_optical, plot_dir, can_plot):
    params = {
        'z': 0.0459,
        'plot': True,
    }
    spex = Spextractor(file_optical, **params)

    spex.create_model(downsampling=3.)

    fig, ax = spex.plot

    name = 'test_prediction'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)


def test_process(file_optical, plot_dir, can_plot):
    params = {
        'z': 0.0459,
        'plot': True,
    }
    spex = Spextractor(file_optical, **params)

    spex.create_model(downsampling=3.)
    spex.process()

    fig, ax = spex.plot

    name = 'test_process'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)


def test_prediction_NIR(file_NIR, plot_dir, can_plot):
    params = {
        'z': 0.001208,
        'plot': True,
        'wave_unit': 'microns',
        'wave_range': (0.4500, 1.5000),
    }
    spex = Spextractor(file_NIR, **params)

    spex.create_model(downsampling=3.)

    features = ('Si II 6150A', 'Si II 5800A')
    spex.process(features)

    fig, ax = spex.plot

    name = 'test_prediction_NIR'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)


def test_prediction_optical_external(file_optical, plot_dir, can_plot):
    params = {
        'z': 0.0459,
        'plot': True,
    }
    spex = Spextractor(file_optical, **params)

    spex.create_model(downsampling=3.)

    # Plot
    fig, ax = basic_spectrum()

    wave = spex.spectrum.wave
    flux = spex.spectrum.flux
    error = spex.spectrum.error

    ax.set_ylim(0., 1.)

    ax.plot(
        wave, flux, color='k', alpha=0.7, lw=1, zorder=0
    )
    ax.fill_between(
        wave, flux - error, flux + error,
        color='grey', alpha=0.5, zorder=-1
    )

    wave_pred = np.linspace(
        spex.spectrum.wave_start, spex.spectrum.wave_end, 2000
    )
    flux_pred, std_pred = spex.predict(wave_pred)

    ax.plot(
        wave_pred, flux_pred,
        color='red', zorder=2, lw=1
    )
    ax.fill_between(
        wave_pred, flux_pred - std_pred, flux_pred + std_pred,
        alpha=0.3, color='red', zorder=1
    )

    name = 'test_prediction_optical_external'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)


def test_prediction_NIR_external(file_NIR, plot_dir, can_plot):
    params = {
        'z': 0.001208,
        'plot': True,
        'wave_unit': 'microns',
        'wave_range': (0.4500, 1.5000),
    }
    spex = Spextractor(file_NIR, **params)

    spex.create_model(downsampling=3.)

    # Plot
    fig, ax = basic_spectrum()

    wave_factor = spex.spectrum.wave_factor
    wave = spex.spectrum.wave / wave_factor
    flux = spex.spectrum.flux
    error = spex.spectrum.error

    ax.set_ylim(0., 1.)

    ax.plot(
        wave, flux, color='k', alpha=0.7, lw=1, zorder=0
    )
    ax.fill_between(
        wave, flux - error, flux + error,
        color='grey', alpha=0.5, zorder=-1
    )

    wave_pred = np.linspace(
        spex.spectrum.wave_start / wave_factor,
        spex.spectrum.wave_end / wave_factor,
        2000
    )
    flux_pred, std_pred = spex.predict(wave_pred)

    ax.plot(
        wave_pred, flux_pred,
        color='red', zorder=2, lw=1
    )
    ax.fill_between(
        wave_pred, flux_pred - std_pred, flux_pred + std_pred,
        alpha=0.3, color='red', zorder=1
    )

    name = 'test_prediction_NIR_external'
    ax.set_title(f'{name}')
    fn = f'{plot_dir}/{name}.png'
    if can_plot:
        fig.savefig(fn)
