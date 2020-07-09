import numpy as np
from pandas import isna
from scipy import interpolate, signal
from sklearn.cluster import DBSCAN, MeanShift
import GPy

import matplotlib.pyplot as plt

import time

from .io import load_spectra
from .downsample import downsample, get_downsample_factor
from .log import setup_log
from .manual import ManualRange
from .lines import get_lines


class Spextractor:

    def __init__(self, data, z=None, remove_gaps=True, SNtype='Ia',
                 manual_range=False, auto_prune=True):

        log_fn = None
        if isinstance(data, str):
            log_fn = f'{data.rsplit(".", 1)[0]:s}.log'
        self.logger = setup_log(log_fn)

        self.wave, self.flux, self.flux_err = self._setup_data(data)
        self._correct_redshift(z)
        self._normalize_flux()

        if remove_gaps:
            self._remove_gaps()

        if isinstance(SNtype, str):
            self.lines = get_lines(SNtype)
        else:
            self.lines = SNtype

        if manual_range:
            self.logger.info('Manually changing feature bounds.')
            m = ManualRange(self.wave, self.flux, self.lines, self.logger)
            self.lines = m.def_lines

        if auto_prune:
            self._auto_prune()
            self._normalize_flux()   # If an error is raised, empty spectrum

        # Instance variables
        self.outlier_downsample_factor = 20

        self.wave_pred = None
        self.mean = None
        self.variance = None
        self.model = None
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

        self.rsi = None

    def _setup_data(self, data):
        """Set up flux (with uncertainty) and wavelength data."""
        # Read data from file if needed
        if isinstance(data, str):
            self.logger.info(f'Loading data from {data:s}\n')
            return load_spectra(data)

        wave = data[:, 0]
        flux = data[:, 1]
        try:
            flux_err = data[:, 2]
        except IndexError:
            msg = 'No flux uncertainties found while reading file.'
            self.logger.warning(msg)
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

    def _remove_gaps(self):
        """Remove zero-flux values."""
        self.wave = self.wave[self.flux != 0]
        self.flux_err = self.flux_err[self.flux != 0]
        self.flux = self.flux[self.flux != 0]

    def _auto_prune(self):
        """Remove data outside feature range (for less computation)."""
        wav_min = min(self.lines[l]['lo_range'][0] for l in self.lines) - 500
        wav_max = max(self.lines[l]['hi_range'][1] for l in self.lines) + 500

        self.flux = self.flux[(wav_min <= self.wave) & (self.wave <= wav_max)]
        self.flux_err = self.flux_err[(wav_min <= self.wave) &
                                      (self.wave <= wav_max)]
        self.wave = self.wave[(wav_min <= self.wave) & (self.wave <= wav_max)]

    def _get_gpy_model(self, x, y, y_err=None, x_pred=None,
                       optimize_noise=False):
        """Calculate the GPy model for given data.

        Uses GPy to determine a Gaussian process model based on given training
        data and optimized hyperparameters.  Returns mean and variance
        prediction at input prediction values.

        Args:
            x (ndarray): Input training set.
            y (ndarray): Output training set.
            y_err (ndarray): Uncertainty in y.
            x_pred (ndarray): Input prediction values. Defaults to 'x', so
                              a change is suggested.
            optimize_noise (ndarray): Optimize single-valued noise parameter.

        Returns:
            mean (ndarray): Prediction of model at given input prediction
                            values.
            variance (ndarray): Variance of model at input pred. values.
            m (GPy.models.GPRegression): Fitted GPy model.
            kernel (GPy.kern): Kernel with optimized hyperparameters.

        """
        kernel = GPy.kern.Matern32(1, lengthscale=300, variance=0.001)

        model_uncertainty = False
        if y_err is not None and np.any(y_err):
            model_uncertainty = True
        else:
            optimize_noise = True
            msg = ('No flux uncertainty detected - optimizing noise parameter.')
            self.logger.info(msg)

        # Add flux errors as noise to kernel
        kern = kernel
        if model_uncertainty:
            diag_vars = y_err**2 * np.eye(len(y_err))
            kern_uncertainty = GPy.kern.Fixed(1, diag_vars)
            kern = kernel + kern_uncertainty
            self.logger.info('Flux error added to GPy kernel')

        # Create model
        m = GPy.models.GPRegression(x[:, np.newaxis], y[:, np.newaxis], kern)
        m['Gaussian.noise.variance'][0] = 0.01

        self.logger.info('Created GP')

        # Optimize model
        if model_uncertainty:
            m['.*fixed.variance'].constrain_fixed()

        if not optimize_noise:
            m.Gaussian_noise.fix(1e-6)

        t0 = time.time()
        m.optimize(optimizer='bfgs')

        self.logger.info(f'Optimised in {time.time() - t0:.2f} s.')
        self.logger.info(m)

        # Predict from model
        if model_uncertainty:
            # Use optimized hyperparameters with original kernel
            kernel.lengthscale = kern.Mat32.lengthscale
            kernel.variance = kern.Mat32.variance

        t0 = time.time()

        if x_pred is None:
            x_pred = self.wave
            self.logger.warning('Predicting at training points')

        mean, var = self._predict(x_pred, m, kernel.copy())

        self.logger.info(f'Predicted in {time.time() - t0:.2f} s.\n')

        return mean, var, m, kernel

    def _predict(self, x_pred, model, kernel):
        mean, var = model.predict(x_pred[:, np.newaxis], kern=kernel)

        return mean.squeeze(), var.squeeze()

    def _filter_outliers(self, sigma_outliers, downsample_method):
        """Attempt to remove sharp lines (teluric, cosmic rays...).

        First applies a heavy downsampling and then discards points that are
        further than 'sigma_outliers' standard deviations.

        """
        t0 = time.time()
        x, y, y_err = downsample(self.wave, self.flux, self.flux_err,
                                 binning=self.outlier_downsample_factor,
                                 method=downsample_method)
        self.logger.info(f'Downsampled in {time.time() - t0:.2f} s.\n')

        mean, var, _m, _k = self._get_gpy_model(x, y, y_err=None, x_pred=None,
                                                optimize_noise=True)

        sigma = np.sqrt(var)
        valid = np.abs(self.flux - mean) < sigma_outliers * sigma

        self.wave = self.wave[valid]
        self.flux = self.flux[valid]
        self.flux_err = self.flux_err[valid]

        msg = f'Auto-removed {len(valid) - valid.sum()} data points'
        self.logger.info(msg)

    def _downsample(self, downsampling, downsample_method):
        """Handle downsampling."""
        if downsampling == 1:
            self.logger.info('Data was not downsampled (binning factor = 1)')
            return

        t0 = time.time()

        n_flux_data = self.flux.shape[0]
        sample_limit = 2300   # Depends on Python memory limits
        if n_flux_data / downsampling > sample_limit:
            downsampling = n_flux_data / sample_limit + 0.1
            msg = (f'Flux array is too large for memory. Downsampling '
                   f'factor increased to {downsampling:.3f}')
            self.logger.warning(msg)
        self.wave, self.flux, self.flux_err = \
            downsample(self.wave, self.flux, self.flux_err,
                       binning=downsampling, method=downsample_method)

        t = time.time() - t0
        msg = (f'Downsampled from {n_flux_data} points with factor of '
               f'{downsampling:.2f} in {t:.2f} s.\n')
        self.logger.info(msg)

    def get_speed(self, lam, lam_err, lam0):
        c = 299.792458   # 10^3 km/s
        l_quot = lam / lam0
        velocity = -c * (l_quot**2 - 1) / (l_quot**2 + 1)
        velocity_err = c * 4 * l_quot / (l_quot**2 + 1)**2 * lam_err / lam0
        return velocity, velocity_err

    def _compute_speed(self, lambda_0, wave_line, flux_line, plot):
        # Pick the strongest
        min_index = flux_line.argmin()
        if min_index == 0 or min_index == flux_line.shape[0] - 1:
            # Feature not found
            return np.nan, np.nan

        lambda_m = wave_line[min_index]
        if plot:
            plt.axvline(lambda_m, color='k', linestyle='--')

        # To estimate the error, we sample possible spectra from the posterior
        # and find the minima.
        samples = self.model.posterior_samples_f(wave_line[:, np.newaxis], 100,
                                                 kern=self.kernel.copy())
        samples = samples.squeeze()
        min_sample_indices = samples.argmin(axis=0)

        # Exclude points at either end
        min_sample_indices = min_sample_indices[1:wave_line.shape[0] - 1]
        if min_sample_indices.size == 0:
            return np.nan, np.nan

        lambda_m_err = np.std(wave_line[min_sample_indices])

        vel, vel_err = self.get_speed(lambda_m, lambda_m_err, lambda_0)
        return vel, vel_err

    def _compute_speed_hv(self, lambda_0, wave_line, flux_line, plot,
                          method='MeanShift'):
        # Pick the strongest
        min_index = flux_line.argmin()
        if min_index == 0 or min_index == flux_line.shape[0] - 1:
            # Feature not found
            return [], [], np.nan, np.nan, [], []

        # To estimate the error, we sample possible spectra from the posterior
        # and find the minima.
        samples = self.model.posterior_samples_f(wave_line[:, np.newaxis], 100,
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
            lambda_m = np.mean(wave_line[min_index])
            lambda_m_err = np.std(wave_line[min_index])

            lambdas.append(lambda_m)
            lambdas_err.append(lambda_m_err)

            this_v, this_v_err = self.get_speed(lambda_m, lambda_m_err,
                                                lambda_0)
            vel_hv.append(this_v)
            vel_hv_err.append(this_v_err)

            if plot:
                plt.vlines(lambda_m, flux_line[min_index] - 0.2,
                           flux_line[min_index] + 0.2, color='k',
                           linestyle='--')
        return lambdas, lambdas_err, vel_hv, vel_hv_err

    def _pEW(self, cont_bounds):
        """Calculate the pEW between two chosen points.

        Args:
            cont_bounds (ndarray): Bounds of feature for pEW calculation. Input
                                   as np.array([x1, x2], [y1, y2])

        Returns:
            pEW (float): Pseudo-equivalent width.
            pEW_err (float): pEW error.

        """
        # Define linear pseudo continuum with cont_bounds
        cont = interpolate.interp1d(cont_bounds[0], cont_bounds[1],
                                    bounds_error=False, fill_value=1)

        # Get ratio of flux within pseudo continuum
        nflux = self.flux / cont(self.wave)
        pEW = 0
        for i in range(len(self.wave)):
            if cont_bounds[0, 0] < self.wave[i] < cont_bounds[0, 1]:
                dwave = 0.5 * (self.wave[i + 1] - self.wave[i - 1])
                pEW += dwave * (1 - nflux[i])

        flux_err = np.abs(signal.cwt(self.flux, signal.ricker, [1])).mean()
        pEW_stat_err = flux_err
        pEW_cont_err = np.abs(cont_bounds[0, 0] - cont_bounds[0, 1]) * flux_err
        pEW_err = np.hypot(pEW_stat_err, pEW_cont_err)

        return pEW, pEW_err

    def _line_depth(self, cont_bounds, wave_line, flux_line, mean_line_err):
        """Calculate line depth for feature

        Args:
            cont_bounds (ndarray): Bounds of feature for pEW calculation. Input
                                   as np.array([x1, x2], [y1, y2])
            feature_min (ndarray): [x, y] point of feature minimum

        Returns:
            depth (float): Depth of line from pseudo continuum

        """
        cont = interpolate.interp1d(cont_bounds[0], cont_bounds[1],
                                    bounds_error=False, fill_value=1)

        min_index = flux_line.argmin()
        if min_index == 0 or min_index == flux_line.shape[0] - 1:
            # Feature not found
            return np.nan

        lambda_m = wave_line[min_index]
        depth = cont(lambda_m) - min(flux_line)
        # the continuum error is already huge so rsi error will honestly be
        # meaningless
        depth_err = mean_line_err[min_index]

        if depth < 0:
            msg = f'Calculated unphysical line depth: {depth:.3f}'
            self.logger.warning(msg)

        return depth, depth_err

    def predict(self, x_pred):
        """Use the created GPR model to make a prediction at any given points.

        Args:
            x_pred (numpy.ndarray): Test input set at which to predict.

        Returns:
            mean (numpy.ndarray): Mean values at the prediction points.
            var (numpy.ndarray): Variance at the prediction points.

        """
        return self._predict(x_pred, self.model, self.kernel)

    def process(self, sigma_outliers=None, downsample_method='weighted',
                downsampling=None, downsampling_R=None, model_uncertainty=True,
                optimize_noise=False, predict_size=2000, plot=False,
                calc_pew_vel=True, high_velocity=False,
                hv_clustering_method='MeanShift'):
        """Run the spectra-fitting and velocity/pEW calculations for this
           object.

        Args:
            sigma_outliers (float): Number of sigma from which to determine
                                    spikes/outliers.
            downsample_method (str): Type of downsampling (weighted, remove).
            downsampling (float): Downsampling factor (>= 1).
            downsampling_R (float): Resolution with which to automatically
                                    determine downsampling factor. This has
                                    priority over 'downsampling' parameter.
            model_uncertainty (bool): Include flux uncertainties in GPy model
                                      inference.
            optimize_noise (bool): Optimize single-valued noise parameter in
                                   GPy model.
            predict_size (int): Prediction input sample size for GPy model.
            plot (bool): Create a plot of data, model, and spectral features.
            calc_pew_vel (bool): Perform velocity/pEW calculations.
            high_velocity (bool): Calculate based on high-velocity properties.
            hv_clustering_method (str): Clustering method for high-velocity
                                        calculations.

        Returns:
            self.pew (dict): pEWs for each feature.
            self.pew_err (dict): pEW uncertainties for each feature.
            self.vel (dict): Velocties for each feature.
            self.vel_err (dict): Velocity uncertainties for each feature.
            self.model (GPy.models.GPRegression): Fitted GPy model.

        """
        ds_methods = ('weighted', 'remove')
        assert downsample_method in ds_methods, \
            f'"downsample_method" must be {ds_methods:s}'

        assert isinstance(predict_size, int), "'predict_size' must be int-valued"

        if optimize_noise and model_uncertainty:
            msg = ('Having a non-zero noise with given uncertainty is not '
                   'statistically legitimate.')
            self.logger.warning(msg)

        t00 = time.time()

        if sigma_outliers is not None:
            self._filter_outliers(sigma_outliers, downsample_method)
            self._normalize_flux()

        if downsampling_R is not None:
            if downsampling is not None:
                msg = ("'downsampling' parameter overridden by "
                       "'downsampling_R' value")
                self.logger.info(msg)
            downsampling = get_downsample_factor(self.wave, downsampling_R)

        if downsampling is not None:
            self._downsample(downsampling, downsample_method)
            # self._normalize_flux()

        self.wave_pred = np.linspace(self.wave[0], self.wave[-1], predict_size)
        y_err = np.zeros_like(self.flux_err)
        if model_uncertainty:
            y_err = self.flux_err
        self.mean, self.variance, self.model, self.kernel = \
            self._get_gpy_model(self.wave, self.flux, y_err=y_err,
                                x_pred=self.wave_pred,
                                optimize_noise=optimize_noise)
        sigma = np.sqrt(self.variance)

        if plot:
            plt.figure()
            plt.xlabel(r"$\mathrm{Rest\ wavelength}\ (\AA)$", size=14)
            plt.ylabel(r"$\mathrm{Normalised\ flux}$", size=14)
            plt.ylim([0.0, 1.0])

            plt.plot(self.wave, self.flux, color='k', alpha=0.5)

            plt.plot(self.wave_pred, self.mean, color='red')
            plt.fill_between(self.wave_pred, self.mean - sigma,
                             self.mean + sigma, alpha=0.3, color='red')

        if not calc_pew_vel:
            return self.model

        t0 = time.time()

        for element in self.lines:
            rest_wavelength = self.lines[element]['rest']
            lo_range = self.lines[element]['lo_range']
            hi_range = self.lines[element]['hi_range']

            index_lo_1, index_lo_2 = np.searchsorted(self.wave_pred, lo_range)
            index_hi_1, index_hi_2 = np.searchsorted(self.wave_pred, hi_range)

            if index_lo_1 == index_lo_2 or index_hi_1 == index_hi_2:
                # Feature outside of range of the spectrum
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

            max_index_lo = index_lo_1 + np.argmax(self.mean[index_lo_1: index_lo_2])
            max_index_hi = index_hi_1 + np.argmax(self.mean[index_hi_1: index_hi_2])

            wave_line = self.wave_pred[max_index_lo:max_index_hi]
            mean_line = self.mean[max_index_lo:max_index_hi]
            mean_line_err = np.sqrt(self.variance[max_index_lo:max_index_hi])

            # Velocity calculation
            plot_vel = True
            if high_velocity:
                vel_out = self._compute_speed_hv(rest_wavelength, wave_line,
                                                 mean_line, plot,
                                                 hv_clustering_method)

                l_hv, l_hv_err, v_hv, v_hv_err = vel_out
                self.lambda_hv[element] = l_hv
                self.lambda_hv_err[element] = l_hv_err
                self.vel_hv[element] = v_hv
                self.vel_hv_err[element] = v_hv_err
                plot_vel = False

            v, v_err = self._compute_speed(rest_wavelength, wave_line,
                                           mean_line, plot_vel)

            self.vel[element] = v
            self.vel_err[element] = v_err

            if np.isnan(v):
                # The feature was not detected, set the PeW to NaN.
                self.pew[element] = np.nan
                self.pew_err[element] = np.nan
                continue

            # PeW calculation
            coords_w = (wave_line[0], wave_line[-1])
            coords_f = (mean_line[0], mean_line[-1])
            cont_coords = np.array((coords_w, coords_f))
            pew, pew_err = self._pEW(cont_coords)
            self.pew[element] = pew
            self.pew_err[element] = pew_err

            if plot:
                plt.scatter(coords_w, coords_f, color='k', s=80)
                _x_pew = np.linspace(*coords_w)
                _dy = coords_f[1] - coords_f[0]
                _dx = coords_w[1] - coords_w[0]
                _m_pew = _dy / _dx
                _y_pew_hi = _m_pew * _x_pew + coords_f[0] - _m_pew * coords_w[0]
                _y_pew_low = self.model.predict(_x_pew[:, None],
                                                kern=self.kernel.copy())[0][:, 0]
                plt.fill_between(_x_pew, _y_pew_low, _y_pew_hi, color='#00a3cc',
                                 alpha=0.3)

            # Line depth (for RSI)
            depth, depth_err = self._line_depth(cont_coords, wave_line,
                                                mean_line, mean_line_err)
            self.line_depth[element] = depth
            self.line_depth_err[element] = depth_err

        msg = f'Velocity and pEW calculations took {time.time() - t0:.3f} s.'
        self.logger.info(msg)

        try:
            ld5800 = self.line_depth['Si II 5800A']
            ld6150 = self.line_depth['Si II 6150A']
            ld5800_err = self.line_depth_err['Si II 5800A']
            ld6150_err = self.line_depth_err['Si II 6150A']
            self.rsi = ld5800 / ld6150
            self.rsi_err = np.sqrt(ld5800_err**2 + ld6150_err**2)
        except KeyError:
            self.rsi = np.nan
            self.rsi_err = np.nan

        self.logger.info(f'Total processing time: {time.time() - t00:.3f}')
        self.logger.info('Complete.')

        self.logger.handlers = []   # Close log handlers between instantiations

        if high_velocity:
            outputs = self.pew, self.pew_err, self.vel, self.vel_err, \
                self.lambda_hv, self.lambda_hv_err, self.vel_hv, \
                self.vel_hv_err, self.model
        else:
            outputs = self.pew, self.pew_err, self.vel, self.vel_err, self.model

        return outputs
