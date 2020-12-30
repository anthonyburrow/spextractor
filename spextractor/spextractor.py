import numpy as np

import time

from .util.io import load_spectra
from .util.log import setup_log
from .util.manual import ManualRange
from .physics.lines import get_lines
from .physics import doppler
from .physics import feature
from .physics.downsample import downsample
from .math import interpolate
from .math import gpr


class Spextractor:

    def __init__(self, data, z=None, sn_type='Ia', manual_range=False,
                 remove_zeroes=True, auto_prune=True, prune_excess=250,
                 outlier_downsampling=20):

        log_fn = None
        if isinstance(data, str):
            log_fn = f'{data.rsplit(".", 1)[0]:s}.log'
        self._logger = setup_log(log_fn)

        self.wave, self.flux, self.flux_err = self._setup_data(data)
        self.wave = doppler.deredshift(self.wave, z=z)
        self._normalize_flux()

        if remove_zeroes:
            self._remove_zeroes()

        if isinstance(sn_type, str):
            self._lines = get_lines(sn_type)
        else:
            self._lines = sn_type

        if manual_range:
            self._logger.info('Manually changing feature bounds...')
            m = ManualRange(self.wave, self.flux, self._lines, self._logger)
            self._lines = m.def_lines

        if auto_prune:
            self._auto_prune(prune_excess)
            self._normalize_flux()

        self._outlier_ds_factor = outlier_downsampling

        # Undefined instance variables
        self._model = None
        self.kernel = None

        self.pew = {}
        self.pew_err = {}
        self.vel = {}
        self.vel_err = {}
        self.lambda_hv = {}
        self.lambda_hv_err = {}
        self.vel_hv = {}
        self.vel_hv_err = {}
        self.line_depth = {}
        self.line_depth_err = {}

        self._fig, self._ax = None, None

    def _setup_data(self, data):
        """Set up flux (with uncertainty) and wavelength data."""
        if isinstance(data, str):
            self._logger.info(f'Loading data from {data:s}\n')
            return load_spectra(data)

        wave = data[:, 0]
        flux = data[:, 1]
        try:
            flux_err = data[:, 2]
        except IndexError:
            msg = 'No flux uncertainties found while reading file.'
            self._logger.info(msg)
            flux_err = np.zeros(len(flux))

        return wave, flux, flux_err

    def _normalize_flux(self):
        """Normalize the flux."""
        max_flux = self.flux.max()
        self.flux /= max_flux

        if np.any(self.flux_err):
            self.flux_err /= max_flux

    def _remove_zeroes(self):
        """Remove zero-flux values."""
        mask = self.flux != 0
        self.wave = self.wave[mask]
        self.flux_err = self.flux_err[mask]
        self.flux = self.flux[mask]

    def _auto_prune(self, prune_excess):
        """Remove data outside feature range (for less computation)."""
        wav_min = min(self._lines[li]['lo_range'][0] for li in self._lines)
        wav_max = max(self._lines[li]['hi_range'][1] for li in self._lines)

        wav_min -= prune_excess
        wav_max += prune_excess

        mask = (wav_min <= self.wave) & (self.wave <= wav_max)

        self.flux = self.flux[mask]
        self.flux_err = self.flux_err[mask]
        self.wave = self.wave[mask]

    def _filter_outliers(self, sigma_outliers, downsample_method):
        """Attempt to remove sharp lines (teluric, cosmic rays...).

        First applies a heavy downsampling and then discards points that are
        further than 'sigma_outliers' standard deviations.

        """
        x, y, _ = downsample(self.wave, self.flux, self.flux_err,
                             binning=self._outlier_ds_factor,
                             method=downsample_method)

        model, kernel = gpr.model(x, y, y_err=None, optimize_noise=True,
                                  logger=self._logger)
        mean, var = gpr.predict(self.wave, model, kernel)

        sigma = np.sqrt(var)
        valid = np.abs(self.flux - mean) < sigma_outliers * sigma

        self.wave = self.wave[valid]
        self.flux = self.flux[valid]
        self.flux_err = self.flux_err[valid]

        msg = (f'Removed {len(valid) - valid.sum()} outliers outside a '
               f'{sigma_outliers}-sigma threshold.')
        self._logger.info(msg)

    def _downsample(self, downsampling, downsample_method):
        """Handle downsampling."""
        n_flux_data = len(self.flux)
        sample_limit = 2300   # Depends on Python memory limits
        if n_flux_data / downsampling > sample_limit:
            downsampling = n_flux_data / sample_limit + 0.1
            msg = (f'Flux array is too large for memory. Downsampling '
                   f'factor increased to {downsampling:.3f}')
            self._logger.warning(msg)
        self.wave, self.flux, self.flux_err = \
            downsample(self.wave, self.flux, self.flux_err,
                       binning=downsampling, method=downsample_method)

        msg = (f'Downsampled from {n_flux_data} points with factor of '
               f'{downsampling:.2f}.\n')
        self._logger.info(msg)

    def _setup_plot(self, gpr_w, gpr_f, gpr_fe):
        if self._fig is not None:
            return

        import matplotlib.pyplot as plt

        self._fig, self._ax = plt.subplots()

        self._ax.set_xlabel(r'$\mathrm{Rest\ wavelength}\ (\AA)$', size=14)
        self._ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', size=14)
        self._ax.set_ylim(0, 1)

        self._ax.plot(self.wave, self.flux, color='k', alpha=0.7, lw=1,
                      zorder=0)
        self._ax.fill_between(self.wave, self.flux - self.flux_err,
                              self.flux + self.flux_err,
                              alpha=0.3, color='grey', zorder=-1)

        self._ax.plot(gpr_w, gpr_f, color='red', zorder=2, lw=1)
        self._ax.fill_between(gpr_w, gpr_f - gpr_fe, gpr_f + gpr_fe,
                              alpha=0.3, color='red', zorder=1)

    def predict(self, X_pred):
        """Use the created GPR model to make a prediction at any given points.

        Args:
            x_pred (numpy.ndarray): Test input set at which to predict.

        Returns:
            mean (numpy.ndarray): Mean values at the prediction points.
            var (numpy.ndarray): Variance at the prediction points.

        """
        return gpr.predict(X_pred, self.model, self.kernel)

    def create_model(self, sigma_outliers=3, downsample_method='weighted',
                     downsampling=None, model_uncertainty=True,
                     optimize_noise=False):
        if optimize_noise and model_uncertainty:
            msg = ('Having a non-zero noise with given uncertainty is not '
                   'statistically legitimate.')
            self._logger.warning(msg)

        if sigma_outliers is not None:
            self._filter_outliers(sigma_outliers, downsample_method)
            self._normalize_flux()

        if downsampling is not None:
            self._downsample(downsampling, downsample_method)
            # self._normalize_flux()

        y_err = np.zeros_like(self.flux_err)
        if model_uncertainty:
            y_err = self.flux_err

        model, kern = gpr.model(self.wave, self.flux, y_err=y_err,
                                optimize_noise=optimize_noise,
                                logger=self._logger)

        self._model = model
        self.kernel = kern

        return model

    @property
    def model(self):
        if self._model is None:
            msg = ('Attempted to use model without generating first. '
                   'Creating model with default parameters...')
            self._logger.warning(msg)
            self.create_model()

        return self._model

    def process(self, high_velocity=False, hv_clustering_method='meanshift',
                plot=False, predict_res=2000):
        """Calculate the line velocities, pEWs, and line depths of each
           feature.

        Parameters
        ----------
        high_velocity : bool, optional
            Calculate based on high-velocity properties. Default is False.
        hv_clustering_method : str, optional
            Clustering method for high-velocity calculations. Can be 'dbscan'
            or 'meanshift'. Default is 'meanshift'.
        plot : bool, optional
            Create a plot of data, model, and spectral features. Default is
            False.
        predict_res : int, optional
            Sample size (resolution) of prediction values predicted by GPy
            model.
        """
        t0 = time.time()

        gpr_wave_pred = np.linspace(self.wave[0], self.wave[-1], predict_res)
        gpr_mean, gpr_variance = self.predict(gpr_wave_pred)
        gpr_sigma = np.sqrt(gpr_variance)

        if plot:
            self._setup_plot(gpr_wave_pred, gpr_mean, gpr_sigma)

        for element in self._lines:
            # Get feature slice
            rest_wave = self._lines[element]['rest']
            lo_range = self._lines[element]['lo_range']
            hi_range = self._lines[element]['hi_range']

            lo_mask = (lo_range[0] <= gpr_wave_pred) & (gpr_wave_pred <= lo_range[1])
            hi_mask = (hi_range[0] <= gpr_wave_pred) & (gpr_wave_pred <= hi_range[1])

            if not (np.any(lo_mask) and np.any(hi_mask)):
                # Feature not contained in spectrum
                if high_velocity:
                    self.lambda_hv[element] = []
                    self.lambda_hv_err[element] = []
                    self.vel_hv[element] = []
                    self.vel_hv_err[element] = []
                self.vel[element] = np.nan
                self.vel_err[element] = np.nan
                self.pew[element] = np.nan
                self.pew_err[element] = np.nan
                continue

            lo_max_ind = gpr_mean[lo_mask].argmax()
            hi_max_ind = gpr_mean[hi_mask].argmax()

            lo_max_wave = gpr_wave_pred[lo_mask][lo_max_ind]
            hi_max_wave = gpr_wave_pred[hi_mask][hi_max_ind]

            mask = (lo_max_wave <= gpr_wave_pred) & (gpr_wave_pred <= hi_max_wave)
            feat_wave = gpr_wave_pred[mask]
            feat_mean = gpr_mean[mask]
            feat_err = np.sqrt(gpr_variance[mask])

            # Velocity calculation
            if high_velocity:
                lam_hv, lam_err_hv, vel_hv, vel_err_hv = \
                    feature.velocity_high(feat_wave, feat_mean, rest_wave,
                                          self._model, self.kernel,
                                          hv_clustering_method)

                self.lambda_hv[element] = lam_hv
                self.lambda_hv_err[element] = lam_err_hv
                self.vel_hv[element] = vel_hv
                self.vel_hv_err[element] = vel_err_hv

            vel, vel_err = feature.velocity(feat_wave, feat_mean, rest_wave,
                                            self._model, self.kernel)

            self.vel[element] = vel
            self.vel_err[element] = vel_err

            if np.isnan(vel):
                self.pew[element] = np.nan
                self.pew_err[element] = np.nan
                self.line_depth[element] = np.nan
                self.line_depth_err[element] = np.nan
                continue

            if plot:
                lam_min = feat_wave[feat_mean.argmin()]
                self._ax.axvline(lam_min, ymax=min(feat_mean), color='k',
                                 linestyle='--')

            # pEW calculation
            pew, pew_err = feature.pEW(feat_wave, feat_mean)
            self.pew[element] = pew
            self.pew_err[element] = pew_err

            if plot:
                wave_range = feat_wave[[0, -1]]
                flux_range = feat_mean[[0, -1]]

                continuum = interpolate.linear(feat_wave, wave_range,
                                               flux_range)

                self._ax.scatter(wave_range, flux_range, color='k', s=30)
                self._ax.fill_between(feat_wave, feat_mean, continuum,
                                      color='#00a3cc', alpha=0.3)

            # Line depth calculation
            line_depth, line_depth_err = feature.depth(feat_wave, feat_mean,
                                                       feat_err)
            self.line_depth[element] = line_depth
            self.line_depth_err[element] = line_depth_err

            if line_depth < 0.:
                msg = (f'Calculated unphysical line depth for {element}:'
                       f'{line_depth:.3f} +- {line_depth_err:.3f}')
                self._logger.warning(msg)

        self._logger.info(f'Calculations took {time.time() - t0:.3f} s.')
        self._logger.handlers = []   # Close log handlers between instantiations

    @property
    def rsi(self):
        try:
            ld5800 = self.line_depth['Si II 5800A']
            ld6150 = self.line_depth['Si II 6150A']
            _rsi = ld5800 / ld6150
        except KeyError:
            _rsi = np.nan

        return _rsi

    @property
    def rsi_err(self):
        try:
            ld5800_err = self.line_depth_err['Si II 5800A']
            ld6150_err = self.line_depth_err['Si II 6150A']
            _rsi_err = np.sqrt(ld5800_err**2 + ld6150_err**2)
        except KeyError:
            _rsi_err = np.nan

        return _rsi_err

    @property
    def plot(self):
        if self._fig is None:
            msg = ('Attempted to call plot property without processing with '
                   '`plot=True`')
            self._logger.error(msg)
            return None, None

        return self._fig, self._ax
