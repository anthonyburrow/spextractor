import numpy as np
import matplotlib.pyplot as plt
import time

from .util.io import load_spectra
from .util.preprocessing import preprocess
from .util.log import setup_log
from .util.manual import ManualRange
from .physics import feature
from .physics.lines import get_features
from .physics.downsample import downsample
from .math import interpolate, gpr


class Spextractor:

    def __init__(self, data, sn_type=None, manual_range=False,
                 outlier_downsampling=20., normalize=True, wave_unit=None,
                 plot=False, log=False, *args, **kwargs):
        """Constructor for the Spextractor class.

        Parameters
        ----------
        data : str, numpy.ndarray
            Spectral data (can be unnormalized) where columns are wavelengths
            (Angstroms), flux, and flux uncertainty.
        sn_type : str, optional
            Type of SN, used to determine which features are able to be
            processed. This must be a key in "./physics/lines.py" ('Ia', 'Ib',
            or 'Ic'). If None, 'Ia' is used by default.
        manual_range : bool, optional
            Used for manually setting feature minimum/maximum ranges via a
            spectrum plot. False by default.
        outlier_downsampling : float, optional
            Downsampling factor if outliers are removed from the GPR training
            set. This is meant to be relatively large for quickly performing
            the first GPR fit used to select outliers.
        normalize : bool, optional
            Determines whether the spectrum should be normalized by maximum
            flux. Default is True (recommended, as GPR will not work well
            otherwise).
        wave_unit : str, optional
            Unit of wavelength for the input data. Available units are
            'angstrom' and 'micron'. If None, this defaults to 'angstrom'.
        plot : bool, optional
            Create and hold a plot of the data, GPR, and velocity/pEW
            information that may be calculated. False by default.
        log : bool, optional
            Determines whether or not a file is created to log console output.

        **kwargs
            z : float, optional
                Redshift value for rest-frame wavelength correction.
            wave_range : tuple, optional
                Manually-set pruning window in Angstroms. Default is None.
            remove_zeroes : bool, optional
                Completely remove data points with flux equal to 0 in the data
                set.
            remove_telluric : bool, optional
                Remove telluric features before correcting for host redshift.
            phot_file : str, optional
                SNooPy-readable file of observed photometry used for
                mangling the spectrum. Note that `time` must also be provided
                to interpolate these light-curves.
            time : float, optional
                Time that spectrum was observed; used for mangling. This may
                be MJD or phase past B maximum time.
            time_format : str, optional
                Format of `time` kwarg given. Default is 'mjd' for MJD time;
                'phase' is also allowed, which indicates `time` given in days
                past B-maximum.
            t_Bmax : str, optional
                Time of B_max in MJD; used only when time_format is 'phase'.
                If `time_format` is 'phase' and a value of `t_Bmax` is not
                provided, a basic Snoopy fit is performed to estimate `t_Bmax`.
            phot_interp : str, optional
                Method of interpolating photometric light-curves. Possible
                methods are the same as scipy.interpolate.interp1d (e.g.
                'linear', 'quadratic', etc.), as well as a custom method
                'powerlaw'. Default is 'quadratic' for assuming interpolation
                near maximum light.
            host_EBV : float, optional
                Host galaxy color excess used for dereddening.
            host_RV : float, optional
                Host reddening vector used for dereddening.
            MW_EBV : float, optional
                MW galaxy color excess used for dereddening.
            MW_RV : float, optional
                MW reddening vector used for dereddening. Default is 3.1.
            verbose : bool, optional
                Display output on console. Default is False.
            log_dir : str, optional
                Directory in which to store log file (if desired). Default
                is a created directory, "./log".
        """
        log_fn = None
        if isinstance(data, str):
            log_fn = f'{data}.log'
        self._logger = setup_log(filename=log_fn, log_to_file=log,
                                 *args, **kwargs)

        self.mangle_bands = None
        self.data = self._setup_data(data, *args, **kwargs)

        self._wave_unit = wave_unit

        # Setup primary plot of (processed but unnormalized) data
        self._plot = plot
        self._fig, self._ax = None, None

        # Scale data
        self._normalize = normalize
        self.fmax_in = self.flux.max()
        self.fmax_out = self.fmax_in
        self._normalize_flux()

        # Define features
        if sn_type is None:
            sn_type = 'Ia'
        self._features = get_features(sn_type)

        if manual_range:
            self._logger.info('Manually changing feature bounds...')
            m = ManualRange(self.data, self._features, self._logger)
            self._features = m.def_lines

        self._outlier_ds_factor = outlier_downsampling

        # Undefined instance attributes
        self._model = None
        self.kernel = None

        self.pew = {}
        self.pew_err = {}
        self.vel = {}
        self.vel_err = {}
        self.depth = {}
        self.depth_err = {}

    def create_model(self, sigma_outliers=3., downsample_method='weighted',
                     downsampling=None, *args, **kwargs):
        """Makes specifications to the GPR model.

        Parameters
        ----------
        sigma_outliers : float, optional
            Sigma factor that defines outliers. Sigma is the calculated
            standard deviation of the heavily downsampled GPR fit. This is 3.
            by default. Set to `None` to cancel this process.
        downsample_method : str, optional
            Method of downsampling. May be 'weighted' (integrates at each bin
            to conserve photon flux) or 'remove' (uses only every
            `downsampling`th point; this is not as accurate but faster than
            'weighted').
        downsampling : float, optional
            Downsampling factor used for the GPR fit. Downsampled data will
            thus be `1 / downsampling` of its original size. If `downsampling
            <= 1.`, the original data will be used.

        Returns
        -------
        GPy.models.GPRegression
            The GPR model that is created.
        """
        if sigma_outliers > 0.:
            self._filter_outliers(sigma_outliers, downsample_method)

        if downsampling is not None:
            self._downsample(downsampling, downsample_method)

        if self._normalize:
            self.fmax_out *= self.flux.max()
        self._normalize_flux()

        model, kern = gpr.model(self.data, logger=self._logger,
                                wave_unit=self._wave_unit)

        self._model = model
        self.kernel = kern

        if self._plot:
            self._setup_plot()

        return model

    def reset_model(self):
        """Clears the current GPR model."""
        self._model = None
        self.kernel = None

    def predict(self, X_pred):
        """Use the created GPR model to make a prediction at any given points.

        Parameters
        ----------
        X_pred : numpy.ndarray
            Test input set at which to predict.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            A tuple consisting of the mean and variance values calculated at
            each of the prediction points.
        """
        mean, var = gpr.predict(X_pred, self.model, self.kernel)

        if self._plot:
            err = np.sqrt(var)
            self._ax.plot(X_pred, mean, color='red', zorder=2, lw=1)
            self._ax.fill_between(X_pred, mean - err, mean + err,
                                  alpha=0.3, color='red', zorder=1)

        return mean, var

    def process(self, features=None, predict_res=2000, hv_features=None,
                high_velocity=True):
        """Calculate the line velocities, pEWs, and line depths of each
           feature.

        Parameters
        ----------
        features : tuple, optional
            Iterable containing strings of features for which to calculate
            properties. These must be included in "./physics/lines.py". By
            default, every feature in "lines.py" is processed.
        predict_res : int, optional
            Sample size (resolution) of prediction values predicted by GPR
            model.
        hv_features : tuple, optional
            Tuple of feature strings for which to use high-velocity wavelength
            ranges.
        high_velocity : bool, optional
            Use high-velocity wavelength ranges for features provided in
            hv_features. This is mostly used as a convenient toggle. Default
            is True.
        """
        t0 = time.time()

        gpr_wave_pred = np.linspace(self.wave[0], self.wave[-1], predict_res)
        gpr_mean, gpr_variance = self.predict(gpr_wave_pred)

        if features is None:
            features = self._features

        if hv_features is None:
            hv_features = []

        for _feature in features:
            rest_wave = self._features[_feature]['rest']

            lo_range = self._features[_feature]['lo_range']
            hi_range = self._features[_feature]['hi_range']
            if high_velocity and _feature in hv_features:
                try:
                    lo_range = self._features[_feature]['lo_range_hv']
                    hi_range = self._features[_feature]['hi_range_hv']
                except KeyError:
                    msg = f'{_feature} does not have defined HV wavelengths'
                    self._logger.warning(msg)

            lo_mask = (lo_range[0] <= gpr_wave_pred) & (gpr_wave_pred <= lo_range[1])
            hi_mask = (hi_range[0] <= gpr_wave_pred) & (gpr_wave_pred <= hi_range[1])

            # If the feature isn't recognized in the spectrum
            if not (np.any(lo_mask) and np.any(hi_mask)):
                self.vel[_feature] = np.nan
                self.vel_err[_feature] = np.nan
                self.pew[_feature] = np.nan
                self.pew_err[_feature] = np.nan
                continue

            lo_max_ind = gpr_mean[lo_mask].argmax()
            hi_max_ind = gpr_mean[hi_mask].argmax()

            lo_max_wave = gpr_wave_pred[lo_mask][lo_max_ind]
            hi_max_wave = gpr_wave_pred[hi_mask][hi_max_ind]

            mask = (lo_max_wave <= gpr_wave_pred) & (gpr_wave_pred <= hi_max_wave)
            feat_data = np.zeros((mask.sum(), 3))
            feat_data[:, 0] = gpr_wave_pred[mask]
            feat_data[:, 1] = gpr_mean[mask]
            feat_data[:, 2] = np.sqrt(gpr_variance[mask])

            # Velocity calculation
            vel, vel_err = feature.velocity(feat_data, rest_wave,
                                            self._model, self.kernel)

            self.vel[_feature] = vel
            self.vel_err[_feature] = vel_err

            if np.isnan(vel):
                self.pew[_feature] = 0.
                self.pew_err[_feature] = 0.
                self.depth[_feature] = 0.
                self.depth_err[_feature] = 0.
                continue

            if self._plot:
                min_ind = feat_data[:, 1].argmin()
                lam_min = feat_data[min_ind, 0]
                self._ax.axvline(lam_min, ymax=feat_data[:, 1].min(),
                                 color='k', linestyle='--')
                self._ax.text(lam_min + 30., 0.015, _feature, rotation=90.,
                              fontsize=8.)

            # pEW calculation
            pew, pew_err = feature.pEW(feat_data)
            self.pew[_feature] = pew
            self.pew_err[_feature] = pew_err

            if self._plot:
                feat_range = feat_data[[0, -1]]
                continuum, _ = interpolate.linear(feat_data[:, 0], feat_range)

                self._ax.scatter(feat_range[:, 0], feat_range[:, 1], color='k',
                                 s=30)
                self._ax.fill_between(feat_data[:, 0], feat_data[:, 1],
                                      continuum, color='#00a3cc', alpha=0.3)

            # Line depth calculation
            depth, depth_err = feature.depth(feat_data)
            self.depth[_feature] = depth
            self.depth_err[_feature] = depth_err

            if depth < 0.:
                msg = (f'Calculated unphysical line depth for {_feature}:'
                       f'{depth:.3f} +- {depth_err:.3f}')
                self._logger.warning(msg)

        self._logger.info(f'Calculations took {time.time() - t0:.3f} s.')
        self._logger.handlers = []   # Close log handlers between instantiations

    @property
    def wave(self):
        return self.data[:, 0]

    @property
    def flux(self):
        return self.data[:, 1]

    @property
    def flux_err(self):
        return self.data[:, 2]

    @property
    def model(self):
        if self._model is None:
            msg = ('Attempted to use model without generating first. '
                   'Creating model with default parameters...')
            self._logger.warning(msg)
            self.create_model()

        return self._model

    @property
    def rsi(self):
        try:
            ld5800 = self.depth['Si II 5800A']
            ld6150 = self.depth['Si II 6150A']
            _rsi = ld5800 / ld6150
        except KeyError:
            _rsi = np.nan

        return _rsi

    @property
    def rsi_err(self):
        try:
            ld5800_err = self.depth_err['Si II 5800A']
            ld6150_err = self.depth_err['Si II 6150A']
            _rsi_err = np.sqrt(ld5800_err**2 + ld6150_err**2)
        except KeyError:
            _rsi_err = np.nan

        return _rsi_err

    @property
    def plot(self):
        return self._fig, self._ax

    def _setup_data(self, data, *args, **kwargs):
        """Set up flux (with uncertainty) and wavelength data."""
        if isinstance(data, str):
            self._logger.info(f'Loading data from {data:s}\n')
            _data = load_spectra(data)
        else:
            _data = data.copy()

        _data = preprocess(_data, spex=self, *args, **kwargs)

        if _data.shape[1] < 3:
            msg = 'No flux uncertainties found.'
            self._logger.info(msg)

            flux_err = np.zeros(len(_data))
            _data = np.c_[_data, flux_err]

        return _data

    def _normalize_flux(self):
        """Normalize the flux."""
        if not self._normalize:
            return

        max_flux = self.flux.max()
        self.data[:, 1] /= max_flux

        if np.any(self.flux_err):
            self.data[:, 2] /= max_flux

    def _filter_outliers(self, sigma_outliers, downsample_method):
        """Attempt to remove sharp lines (teluric, cosmic rays...).

        First applies a heavy downsampling and then discards points that are
        further than 'sigma_outliers' standard deviations.

        """
        ds_data = downsample(self.data, binning=self._outlier_ds_factor,
                             method=downsample_method)

        model, kernel = gpr.model(ds_data, logger=self._logger,
                                  wave_unit=self._wave_unit)
        mean, var = gpr.predict(self.wave, model, kernel)

        sigma = np.sqrt(var)
        valid = np.abs(self.flux - mean) < sigma_outliers * sigma

        self.data = self.data[valid]

        msg = (f'Removed {len(valid) - valid.sum()} outliers outside a '
               f'{sigma_outliers}-sigma threshold.')
        self._logger.info(msg)

    def _downsample(self, downsampling, downsample_method):
        """Handle downsampling."""
        n_points = len(self.flux)
        sample_limit = 5000   # Depends on Python memory limits

        if n_points / downsampling > sample_limit:
            downsampling = n_points / sample_limit + 0.1
            msg = (f'Flux array is too large for memory. Downsampling '
                   f'factor increased to {downsampling:.3f}')
            self._logger.warning(msg)

        self.data = downsample(self.data, binning=downsampling,
                               method=downsample_method)

        msg = (f'Downsampled from {n_points} to {len(self.flux)} points '
               f'with a factor of {downsampling:.2f}.\n')
        self._logger.info(msg)

    def _setup_plot(self):
        """Setup the spectrum plot."""
        if self._fig is not None:
            return

        self._fig, self._ax = plt.subplots()

        self._ax.set_xlabel(r'$\mathrm{Rest\ wavelength}\ (\AA)$', size=14)
        self._ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', size=14)
        self._ax.set_ylim(0, 1)

        self._ax.plot(self.wave, self.flux, color='k', alpha=0.7, lw=1,
                      zorder=0)
        self._ax.fill_between(self.wave, self.flux - self.flux_err,
                              self.flux + self.flux_err, color='grey',
                              alpha=0.5, zorder=-1)
