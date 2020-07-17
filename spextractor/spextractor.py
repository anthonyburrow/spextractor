import numpy as np
from pandas import isna
from scipy import interpolate, signal
from sklearn.cluster import DBSCAN, MeanShift
import GPy

import time

from .io import load_spectra
from .downsample import downsample, get_downsample_factor
from .log import setup_log
from .manual import ManualRange
from .lines import get_lines


class Spextractor:

    def __init__(self, data, z=None, SNtype='Ia', manual_range=False,
                 remove_zeroes=True, auto_prune=True, prune_excess=250,
                 outlier_downsampling=20):

        log_fn = None
        if isinstance(data, str):
            log_fn = f'{data.rsplit(".", 1)[0]:s}.log'
        self._logger = setup_log(log_fn)

        self.wave, self.flux, self.flux_err = self._setup_data(data)
        self._correct_redshift(z)
        self._normalize_flux()

        if remove_zeroes:
            self._remove_zeroes()

        if isinstance(SNtype, str):
            self._lines = get_lines(SNtype)
        else:
            self._lines = SNtype

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
            self._logger.warning(msg)
            flux_err = np.zeros(len(flux))

        return wave, flux, flux_err

    def _correct_redshift(self, z=None):
        """Correct for redshift of host galaxy."""
        if z is not None and not isna(z):
            self.wave /= (1 + z)

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

    def _get_gpr_model(self, x, y, y_err=None, optimize_noise=False):
        """Calculate the GPy model for given data.

        Uses GPy to determine a Gaussian process model based on given training
        data and optimized hyperparameters.

        Parameters
        ----------
        x : numpy.ndarray
            Input training set.
        y : numpy.ndarray
            Output training set.
        y_err : numpy.ndarray
            Uncertainty in observed output `y`.
        optimize_noise : numpy.ndarray
            Optimize single-valued noise parameter.

        Returns
        -------
        m : GPy.models.GPRegression
            Fitted GPy model.
        kernel : GPy.kern
            Kernel with optimized hyperparameters.

        """
        kernel = GPy.kern.Matern32(1, lengthscale=300, variance=0.001)

        model_uncertainty = False
        if y_err is not None and np.any(y_err):
            model_uncertainty = True
        else:
            optimize_noise = True
            msg = ('No flux uncertainty detected - optimizing noise parameter.')
            self._logger.info(msg)

        # Add flux errors as noise to kernel
        kern = kernel
        if model_uncertainty:
            diag_vars = y_err**2 * np.eye(len(y_err))
            kern_uncertainty = GPy.kern.Fixed(1, diag_vars)
            kern = kernel + kern_uncertainty
            self._logger.info('Flux error added to GPy kernel')

        # Create model
        m = GPy.models.GPRegression(x[:, np.newaxis], y[:, np.newaxis], kern)
        m['Gaussian.noise.variance'][0] = 0.01

        self._logger.info('Created GP')

        # Optimize model
        if model_uncertainty:
            m['.*fixed.variance'].constrain_fixed()

        if not optimize_noise:
            m.Gaussian_noise.fix(1e-6)

        t0 = time.time()
        m.optimize(optimizer='bfgs')

        self._logger.info(f'Optimised in {time.time() - t0:.2f} s.')
        self._logger.info(m)

        if model_uncertainty:
            # Use optimized hyperparameters with original kernel
            kernel.lengthscale = kern.Mat32.lengthscale
            kernel.variance = kern.Mat32.variance

        return m, kernel

    def _predict(self, x_pred, model, kernel, verbose=True):
        t0 = time.time()

        mean, var = model.predict(x_pred[:, np.newaxis], kern=kernel.copy())

        if verbose:
            self._logger.info(f'Predicted in {time.time() - t0:.2f} s.\n')

        return mean.squeeze(), var.squeeze()

    def _filter_outliers(self, sigma_outliers, downsample_method):
        """Attempt to remove sharp lines (teluric, cosmic rays...).

        First applies a heavy downsampling and then discards points that are
        further than 'sigma_outliers' standard deviations.

        """
        t0 = time.time()
        x, y, y_err = downsample(self.wave, self.flux, self.flux_err,
                                 binning=self._outlier_ds_factor,
                                 method=downsample_method)
        self._logger.info(f'Downsampled in {time.time() - t0:.2f} s.\n')

        _m, _k = self._get_gpr_model(x, y, y_err=None, optimize_noise=True)
        mean, var = self._predict(self.wave, _m, _k)

        sigma = np.sqrt(var)
        valid = np.abs(self.flux - mean) < sigma_outliers * sigma

        self.wave = self.wave[valid]
        self.flux = self.flux[valid]
        self.flux_err = self.flux_err[valid]

        msg = f'Auto-removed {len(valid) - valid.sum()} data points'
        self._logger.info(msg)

    def _downsample(self, downsampling, downsample_method):
        """Handle downsampling."""
        if downsampling == 1:
            self._logger.info('Data was not downsampled (binning factor = 1)')
            return

        t0 = time.time()

        n_flux_data = self.flux.shape[0]
        sample_limit = 2300   # Depends on Python memory limits
        if n_flux_data / downsampling > sample_limit:
            downsampling = n_flux_data / sample_limit + 0.1
            msg = (f'Flux array is too large for memory. Downsampling '
                   f'factor increased to {downsampling:.3f}')
            self._logger.warning(msg)
        self.wave, self.flux, self.flux_err = \
            downsample(self.wave, self.flux, self.flux_err,
                       binning=downsampling, method=downsample_method)

        t = time.time() - t0
        msg = (f'Downsampled from {n_flux_data} points with factor of '
               f'{downsampling:.2f} in {t:.2f} s.\n')
        self._logger.info(msg)

    def _get_velocity(self, lam, lam_err, lam0):
        c = 299.792458   # 10^3 km/s
        l_quot = lam / lam0
        velocity = -c * (l_quot**2 - 1) / (l_quot**2 + 1)
        velocity_err = c * 4 * l_quot / (l_quot**2 + 1)**2 * lam_err / lam0
        return velocity, velocity_err

    def _compute_speed(self, lambda_0, feat_wave, feat_mean, plot):
        min_index = feat_mean.argmin()
        if min_index == 0 or min_index == feat_mean.shape[0] - 1:
            # Feature not found
            return np.nan, np.nan

        lambda_m = feat_wave[min_index]

        # To estimate the error, we sample possible spectra from the posterior
        # and find the minima.
        samples = self._model.posterior_samples_f(feat_wave[:, np.newaxis], 100,
                                                  kern=self.kernel.copy())
        samples = samples.squeeze()
        min_sample_indices = samples.argmin(axis=0)

        # Exclude points at either end
        min_sample_indices = min_sample_indices[1:feat_wave.shape[0] - 1]
        if min_sample_indices.size == 0:
            return np.nan, np.nan

        lambda_m_err = np.std(feat_wave[min_sample_indices])

        vel, vel_err = self._get_velocity(lambda_m, lambda_m_err, lambda_0)

        if plot:
            self._ax.axvline(lambda_m, ymax=min(feat_mean), color='k',
                             linestyle='--')

        return vel, vel_err

    def _compute_speed_hv(self, lambda_0, feat_wave, feat_mean, plot,
                          method='MeanShift'):
        min_index = feat_mean.argmin()
        if min_index == 0 or min_index == feat_mean.shape[0] - 1:
            # Feature not found
            return [], [], np.nan, np.nan, [], []

        # To estimate the error, we sample possible spectra from the posterior
        # and find the minima.
        samples = self._model.posterior_samples_f(feat_wave[:, np.newaxis], 100,
                                                  kern=self.kernel.copy())
        samples = samples.squeeze()

        minima_samples = []
        for i in range(samples.shape[1]):
            positions = signal.argrelmin(samples[:, i], order=10)[0]
            minima_samples.extend(positions)

        minima_samples = np.array(minima_samples)[:, np.newaxis]

        method = method.lower()
        methods = ('dbscan', 'meanshift')
        assert method in methods, \
            f'Invalid method {method}, valid are MeanShift and DBSCAN'

        if method == 'dbscan':
            labels = DBSCAN(eps=1).fit_predict(minima_samples)
        elif method == 'meanshift':
            labels = MeanShift(10).fit_predict(minima_samples)

        lambdas = []
        lambdas_err = []
        vel_hv = []
        vel_hv_err = []

        for x in np.unique(labels):
            if x == -1:
                # This is the "noise" label in DBSCAN
                continue

            matching = labels == x
            if matching.sum() < 5:
                continue  # This is just noise

            min_index = minima_samples[matching]
            lambda_m = np.mean(feat_wave[min_index])
            lambda_m_err = np.std(feat_wave[min_index])

            lambdas.append(lambda_m)
            lambdas_err.append(lambda_m_err)

            this_v, this_v_err = self._get_velocity(lambda_m, lambda_m_err,
                                                    lambda_0)
            vel_hv.append(this_v)
            vel_hv_err.append(this_v_err)

            if plot:
                self._ax.vlines(lambda_m, feat_mean[min_index] - 0.2,
                                feat_mean[min_index] + 0.2, color='k',
                                linestyle='--')
        return lambdas, lambdas_err, vel_hv, vel_hv_err

    def _compute_pEW(self, feat_wave, feat_mean, plot):
        wave_range = np.array([feat_wave[0], feat_wave[-1]])
        flux_range = np.array([feat_mean[0], feat_mean[-1]])

        # Define linear pseudo-continuum with bounds
        continuum = interpolate.interp1d(wave_range, flux_range,
                                         bounds_error=False, fill_value=1)

        # Integrate the fractional flux
        mask = (wave_range[0] < self.wave) & (self.wave < wave_range[1])
        frac_flux = self.flux[mask] / continuum(self.wave[mask])
        lower_wave = self.wave[:-1][mask[1:]]
        upper_wave = self.wave[1:][mask[:-1]]
        pEW = np.sum(0.5 * (upper_wave - lower_wave) * (1 - frac_flux))

        pEW_stat_err = np.abs(signal.cwt(self.flux, signal.ricker, [1])).mean()
        pEW_cont_err = (wave_range[1] - wave_range[0]) * pEW_stat_err
        pEW_err = np.hypot(pEW_stat_err, pEW_cont_err)

        if plot:
            feat_x = np.linspace(*wave_range)
            cont_dy = flux_range[1] - flux_range[0]
            cont_dx = wave_range[1] - wave_range[0]
            cont_m = cont_dy / cont_dx
            cont_y_hi = cont_m * (feat_x - wave_range[0]) + flux_range[0]
            cont_y_lo, _ = self.predict(feat_x, verbose=False)

            self._ax.scatter(wave_range, flux_range, color='k', s=30)
            self._ax.fill_between(feat_x, cont_y_lo, cont_y_hi,
                                  color='#00a3cc', alpha=0.3)

        return pEW, pEW_err

    def _compute_depth(self, feat_wave, feat_mean, feat_mean_err):
        """Calculate line depth for feature

        Args:
            cont_bounds (ndarray): Bounds of feature for pEW calculation. Input
                                   as np.array([x1, x2], [y1, y2])
            feature_min (ndarray): [x, y] point of feature minimum

        Returns:
            depth (float): Depth of line from pseudo continuum

        """
        wave_range = np.array([feat_wave[0], feat_wave[-1]])
        flux_range = np.array([feat_mean[0], feat_mean[-1]])

        # Define linear pseudo-continuum with bounds
        continuum = interpolate.interp1d(wave_range, flux_range,
                                         bounds_error=False, fill_value=1)

        min_index = feat_mean.argmin()
        if min_index == 0 or min_index == feat_mean.shape[0] - 1:
            # Feature not found
            return np.nan

        lambda_m = feat_wave[min_index]
        depth = continuum(lambda_m) - feat_mean[min_index]
        # the continuum error is already huge so rsi error is meaningless
        depth_err = feat_mean_err[min_index]

        if depth < 0:
            msg = f'Calculated unphysical line depth: {depth:.3f}'
            self._logger.warning(msg)

        return depth, depth_err

    def _setup_plot(self, gpr_w, gpr_f, gpr_fe):
        if self._fig is not None:
            return

        import matplotlib.pyplot as plt

        self._fig, self._ax = plt.subplots()

        self._ax.set_xlabel(r'$\mathrm{Rest\ wavelength}\ (\AA)$', size=14)
        self._ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', size=14)
        self._ax.set_ylim(0, 1)

        self._ax.plot(self.wave, self.flux, color='k', alpha=0.7)
        self._ax.plot(gpr_w, gpr_f, color='red')
        self._ax.fill_between(gpr_w, gpr_f - gpr_fe, gpr_f + gpr_fe,
                              alpha=0.3, color='red')

    def predict(self, x_pred, verbose=True):
        """Use the created GPR model to make a prediction at any given points.

        Args:
            x_pred (numpy.ndarray): Test input set at which to predict.

        Returns:
            mean (numpy.ndarray): Mean values at the prediction points.
            var (numpy.ndarray): Variance at the prediction points.

        """
        return self._predict(x_pred, self.model, self.kernel, verbose)

    def create_model(self, sigma_outliers=3, downsample_method='weighted',
                     downsampling=None, downsampling_R=None,
                     model_uncertainty=True, optimize_noise=False):
        if optimize_noise and model_uncertainty:
            msg = ('Having a non-zero noise with given uncertainty is not '
                   'statistically legitimate.')
            self._logger.warning(msg)

        if sigma_outliers is not None:
            self._filter_outliers(sigma_outliers, downsample_method)
            self._normalize_flux()

        if downsampling_R is not None:
            if downsampling is not None:
                msg = ("'downsampling' parameter overridden by "
                       "'downsampling_R' value")
                self._logger.info(msg)
            downsampling = get_downsample_factor(self.wave, downsampling_R)

        if downsampling is not None:
            self._downsample(downsampling, downsample_method)
            # self._normalize_flux()

        y_err = np.zeros_like(self.flux_err)
        if model_uncertainty:
            y_err = self.flux_err

        model, kern = self._get_gpr_model(self.wave, self.flux, y_err=y_err,
                                          optimize_noise=optimize_noise)

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

    def process(self, high_velocity=False, hv_clustering_method='MeanShift',
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
            feat_mean_err = np.sqrt(gpr_variance[mask])

            # Velocity calculation
            if high_velocity:
                vel_out = self._compute_speed_hv(rest_wave, feat_wave,
                                                 feat_mean, plot,
                                                 hv_clustering_method)

                l_hv, l_hv_err, v_hv, v_hv_err = vel_out
                self.lambda_hv[element] = l_hv
                self.lambda_hv_err[element] = l_hv_err
                self.vel_hv[element] = v_hv
                self.vel_hv_err[element] = v_hv_err

            v, v_err = self._compute_speed(rest_wave, feat_wave, feat_mean,
                                           plot=not high_velocity)

            self.vel[element] = v
            self.vel_err[element] = v_err

            if np.isnan(v):
                # The feature was not detected
                self.pew[element] = np.nan
                self.pew_err[element] = np.nan
                continue

            # pEW calculation
            pew, pew_err = self._compute_pEW(feat_wave, feat_mean, plot)
            self.pew[element] = pew
            self.pew_err[element] = pew_err

            # Line depth calculation
            depth, depth_err = self._compute_depth(feat_wave, feat_mean,
                                                   feat_mean_err)
            self.line_depth[element] = depth
            self.line_depth_err[element] = depth_err

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
