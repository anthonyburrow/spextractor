import time

import matplotlib.pyplot as plt
import numpy as np
from SpectrumCore import Spectrum

from spextractor import Spextractor


def test_model_performance_and_predictions(file_optical, plot_dir):
    """
    Compare fit() timing and predictions for GPR, Spline, and SKI models.

    Generates:
    - Timing report in console.
    - Zoomed plot (5500–6500 Å) showing model predictions and Si II 6150A
      velocity markers.
    """
    data_file = file_optical
    z = 0.0459

    model_types = ['gpr', 'spline', 'ski']
    fit_times = {}
    velocities = {}

    wave_range = (5500.0, 6500.0)

    wave_pred = np.linspace(wave_range[0], wave_range[1], 500)

    fig, ax = plt.subplots()

    ax.set_xlim(*wave_range)
    ax.set_xlabel('Rest-frame Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux (arb. units)', fontsize=12)
    ax.grid(True, alpha=0.3)

    spectrum = Spectrum(data_file, z=z)
    spectrum.deredshift(z=z)
    spectrum.normalize_flux()

    mask = (spectrum.wave >= wave_range[0]) & (spectrum.wave <= wave_range[1])
    ax.plot(
        spectrum.wave[mask],
        spectrum.flux[mask],
        'k-',
        linewidth=1,
        label='Original data',
    )

    spex_params = {
        'z': z,
        'plot': False,
        'log': False,
    }

    print('\n' + '=' * 70)
    print('MODEL PERFORMANCE COMPARISON')
    print('=' * 70)

    for model_type in model_types:
        print(f'\n--- Testing {model_type.upper()} model ---')

        spex = Spextractor(file_optical, **spex_params)

        start_time = time.time()
        spex.create_model(model_type=model_type)
        fit_time = time.time() - start_time

        assert spex._model is not None, (
            f'Model creation failed for {model_type}'
        )

        fit_times[model_type] = fit_time
        print(f'Fit time: {fit_time:.4f} seconds')

        pred_result = spex._model.predict(wave_pred)
        if isinstance(pred_result, tuple):
            y_pred, _ = pred_result
        else:
            y_pred = pred_result
            _ = np.zeros_like(y_pred)

        ax.plot(wave_pred, y_pred, label=model_type.upper())

        spex.process(features=['Si II 6150A'])
        vel = spex.vel.get('Si II 6150A', None)
        if vel is not None:
            velocities[model_type] = vel
            print(f'Si II 6150A velocity: {vel:.2f} km/s')
        else:
            print('Si II 6150A velocity: Not calculated')

    print('\n' + '=' * 70)
    print('TIMING SUMMARY')
    print('=' * 70)
    for model_type in model_types:
        t = fit_times[model_type]
        print(f'{model_type.upper():8s}: {t:.4f} s')

    if fit_times:
        fastest = min(fit_times, key=fit_times.get)  # type: ignore[arg-type]
        slowest = max(fit_times, key=fit_times.get)  # type: ignore[arg-type]
        speedup = fit_times[slowest] / fit_times[fastest]
        print(
            f'\nSpeedup ({slowest.upper()} '
            f'vs {fastest.upper()}): {speedup:.2f}x'
        )

    print('\n' + '=' * 70)
    print('VELOCITIES')
    print('=' * 70)
    for model_type in model_types:
        if model_type in velocities:
            print(
                f'{model_type.upper():8s}: {velocities[model_type]:8.2f} km/s'
            )
        else:
            print(f'{model_type.upper():8s}: Not calculated')

    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()

    file_name = f'{plot_dir}/test_model_comparison.png'
    fig.savefig(file_name, dpi=125)
    print(f"\nPlot saved as '{file_name}'")

    plt.close()
