from collections.abc import Callable

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from SpectrumCore import Spectrum

from ..math.functions import gaussian
from . import doppler


class Feature:
    def __init__(
        self,
        name: str,
        rest_wave: float,
        spectrum: Spectrum,
        gpr_model: GaussianProcessRegressor,
    ) -> None:
        self.name = name
        self.rest_wave = rest_wave

        self.spectrum = spectrum
        self.gpr_model = gpr_model

        self.wave_left: float | None = None
        self.wave_right: float | None = None

        self.feature_data: np.ndarray | None = None

    def update_endpoints(
        self, lo_range: tuple[float, float], hi_range: tuple[float, float]
    ) -> None:
        left_data = self.spectrum.between(lo_range)
        left_ind = left_data[:, 1].argmax()
        self.wave_left = left_data[left_ind, 0]

        right_data = self.spectrum.between(hi_range)
        right_ind = right_data[:, 1].argmax()
        self.wave_right = right_data[right_ind, 0]

        self.reset_feature_data()

    def reset_feature_data(self) -> None:
        if self.wave_left is None or self.wave_right is None:
            raise RuntimeError('Feature endpoints not set.')

        self.feature_data = self.spectrum.between(
            (self.wave_left, self.wave_right)
        )

    def velocity(
        self, velocity_method: str | None = None, *args, **kwargs
    ) -> tuple[float, float, tuple[float, float]]:
        if velocity_method is None:
            velocity_method = 'minimum'

        if velocity_method == 'minimum':
            return self._velocity_minimum(*args, **kwargs)
        elif velocity_method == 'blue_edge':
            return self._velocity_blue_edge(*args, **kwargs)

        raise RuntimeError('Invalid velocity method given')

    def _velocity_minimum(
        self, n_samples: int = 100, *args, **kwargs
    ) -> tuple[float, float, tuple[float, float]]:
        if self.feature_data is None:
            raise RuntimeError('Feature data not set.')

        wave = self.feature_data[:, 0]
        flux = self.feature_data[:, 1]

        min_ind = flux.argmin()

        # If clear feature not found
        if min_ind == 0 or min_ind == len(flux) - 1:
            return np.nan, np.nan, (np.nan, np.nan)

        lam_min = wave[min_ind]

        # To estimate the error, sample possible spectra from the posterior
        # - For some reason, sample_y outputs shape (N_wave, N_samples=100)
        samples = self.gpr_model.sample_y(wave[:, np.newaxis], n_samples)
        min_sample_indices = samples.argmin(axis=0)

        # Exclude points at either end
        min_sample_indices = min_sample_indices[1:-1]
        if len(min_sample_indices) == 0:
            return np.nan, np.nan, (np.nan, np.nan)

        lam_min_err = np.std(wave[min_sample_indices]).astype(float)

        vel, vel_err = doppler.velocity(lam_min, lam_min_err, self.rest_wave)

        draw_point: tuple[float, float] = lam_min, flux.min()

        return vel, vel_err, draw_point

    def _velocity_blue_edge(
        self,
        n_samples: int = 100,
        feat_profile: Callable | None = None,
        profile_params: tuple[float] | None = None,
        *args,
        **kwargs,
    ) -> tuple[float, float, tuple[float, float]]:
        if self.feature_data is None:
            raise RuntimeError('Feature data not set.')

        wave = self.feature_data[:, 0]
        flux = self.feature_data[:, 1]

        if feat_profile is None:
            feat_profile = gaussian

        if profile_params is None:
            mu = wave[flux.argmin()]
            sigma = 0.5 * (wave[-1] - wave[0])
            profile_params = (mu, sigma, -0.1, 0.5)
            bounds = (
                (wave[0], 0.0, -2.0, 0.0),
                (wave[-1], 5.0 * sigma, 0.0, 1.0),
            )

        params, _ = curve_fit(
            feat_profile, wave, flux, p0=profile_params, bounds=bounds
        )
        mu, sigma = params[:2]
        lam = mu - 3.0 * sigma

        # Calculate lambda error through sampling
        samples = self.gpr_model.sample_y(wave[:, np.newaxis], n_samples).T

        lam_samples = []
        for sample in samples:
            params_err, _ = curve_fit(
                feat_profile, wave, sample, p0=profile_params, bounds=bounds
            )
            mu_err, sigma_err = params_err[:2]
            lam_samples.append(mu_err - 3.0 * sigma_err)

        lam_err = float(np.std(lam_samples))

        # Calculate velocity
        vel, vel_err = doppler.velocity(lam, lam_err, self.rest_wave)

        # if spex._plot:
        #     spex._ax.plot(wave, feat_profile(wave, *params), 'b-')

        draw_point = lam, feat_profile(lam, *params)

        return vel, vel_err, draw_point

    def pEW(
        self, n_samples: int = 100, *args, **kwargs
    ) -> tuple[float, float]:
        if self.feature_data is None:
            raise RuntimeError('Feature data not set.')

        wave = self.feature_data[:, 0]
        flux = self.feature_data[:, 1]
        endpoints = self.feature_data[[0, -1], :2]

        continuum = np.interp(wave, endpoints[:, 0], endpoints[:, 1])
        frac_flux = 1.0 - flux / continuum
        pEW = trapezoid(frac_flux, x=wave)

        # For some reason, sample_y outputs shape (N_wave, N_samples=100)
        samples = self.gpr_model.sample_y(wave[:, np.newaxis], n_samples).T
        frac_flux = 1.0 - samples / continuum
        pEW_err = trapezoid(frac_flux, x=wave, axis=1).std()

        # Technically you should also come up with some sort of continuum
        # error, then translate this into frac_flux error, then make this into
        # an integrated pEW error for the continuum, then add in quadrature.
        # Since continuum is unknown in the first place, this is just not worth
        # it. Continuum is therefore assumed as rigid.

        return pEW, pEW_err
