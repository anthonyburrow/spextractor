import numpy as np
import matplotlib.pyplot as plt
import time

from SpectrumCore.Spectrum import Spectrum

from .util.log import setup_log
from .util.manual import ManualRange
from .physics import feature
from .physics.lines import get_features
from .math import interpolate, gpr


class Spextractor:

    def __init__(self, data, sn_type=None, manual_range=False, wave_unit=None,
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

        self.spectrum = Spectrum(data)
        self._preprocess_spectrum(*args, **kwargs)

        self._wave_unit = wave_unit

        # Setup primary plot of (processed but unnormalized) data
        self._plot = plot
        self._fig, self._ax = None, None

        # Scale data
        # self._normalize = normalize
        # self.fmax_in = self.flux.max()
        # self.fmax_out = self.fmax_in
        # self._normalize_flux()

        # Define features
        if sn_type is None:
            sn_type = 'Ia'
        self._features = get_features(sn_type)

        if manual_range:
            self._logger.info('Manually changing feature bounds...')
            m = ManualRange(self.spectrum.data, self._features, self._logger)
            self._features = m.def_lines

        # Undefined instance attributes
        self._model = None
        self.kernel = None

        self.pew = {}
        self.pew_err = {}
        self.vel = {}
        self.vel_err = {}
        self.depth = {}
        self.depth_err = {}

    def create_model(self, downsampling=None, *args, **kwargs):
        """Makes specifications to the GPR model.

        Parameters
        ----------
        downsampling : float, optional
            Downsampling factor used for the GPR fit. Downsampled data will
            thus be `1 / downsampling` of its original size. If `downsampling
            <= 1.`, the original data will be used.

        Returns
        -------
        GPy.models.GPRegression
            The GPR model that is created.
        """
        if downsampling is None:
            downsampling = 1.
        self._downsample(downsampling)

        model, kern = gpr.model(
            self.spectrum.data, logger=self._logger, wave_unit=self._wave_unit
        )

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
            self._ax.fill_between(
                X_pred, mean - err, mean + err,
                alpha=0.3, color='red', zorder=1
            )

        return mean, var

    def process(self, features=None, predict_res=2000, *args, **kwargs):
        """Calculate the line velocities, pEWs, and line depths of each
           feature.

        Parameters
        ----------
        features : tuple, optional
            Iterable containing strings of features for which to calculate
            properties. These must be included in "./physics/lines.py". By
            default, every feature in "lines.py" is processed.
        predict_res : int, optional
            Sample size (resolution) of values predicted by GPR model.

        **kwargs:
            velocity_method
        """
        t0 = time.time()

        gpr_wave_pred = np.linspace(
            self.spectrum.wave_start, self.spectrum.wave_end, predict_res
        )
        gpr_mean, gpr_variance = self.predict(gpr_wave_pred)

        if features is None:
            features = self._features

        for _feature in features:
            rest_wave = self._features[_feature]['rest']

            lo_range = self._features[_feature]['lo_range']
            hi_range = self._features[_feature]['hi_range']

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
            vel, vel_err, draw_point = \
                feature.velocity(feat_data, rest_wave, self, *args, **kwargs)

            self.vel[_feature] = vel
            self.vel_err[_feature] = vel_err

            if np.isnan(vel):
                self.pew[_feature] = 0.
                self.pew_err[_feature] = 0.
                self.depth[_feature] = 0.
                self.depth_err[_feature] = 0.
                continue

            if self._plot:
                self._ax.axvline(
                    draw_point[0], ymax=draw_point[1],
                    color='k', linestyle='--'
                )
                self._ax.text(
                    draw_point[0] + 30., 0.015, _feature,
                    rotation=90., fontsize=8.
                )

            # pEW calculation
            pew, pew_err = feature.pEW(feat_data)
            self.pew[_feature] = pew
            self.pew_err[_feature] = pew_err

            if self._plot:
                feat_range = feat_data[[0, -1]]
                continuum, _ = interpolate.linear(feat_data[:, 0], feat_range)

                self._ax.scatter(
                    feat_range[:, 0], feat_range[:, 1], color='k', s=30
                )
                self._ax.fill_between(
                    feat_data[:, 0], feat_data[:, 1], continuum,
                    color='#00a3cc', alpha=0.3
                )

            # Line depth calculation
            depth, depth_err = feature.depth(feat_data)
            self.depth[_feature] = depth
            self.depth_err[_feature] = depth_err

            if depth < 0.:
                msg = (
                    f'Calculated unphysical line depth for {_feature}:'
                    f'{depth:.3f} +- {depth_err:.3f}'
                )
                self._logger.warning(msg)

        self._logger.info(f'Calculations took {time.time() - t0:.3f} s.')
        self._logger.handlers = []   # Close log handlers between instantiations

    @property
    def model(self):
        if self._model is None:
            msg = (
                'Attempted to use model without generating first. '
                'Creating model with default arguments...'
            )
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

    def _preprocess_spectrum(
        self, z: float = None, wave_range: tuple[float] = None,
        host_EBV=None, host_RV=None, MW_EBV=None, MW_RV=3.1,
        *args, **kwargs
    ):
        self.spectrum.remove_nans()
        self.spectrum.remove_nonpositive()

        self.spectrum.remove_telluric()   # TODO: AFTER MANGLING

        if z is not None and z != 0.:
            self.spectrum.deredshift(z)

        if wave_range is not None:
            self.spectrum.prune(wave_range)

        # Milky Way extinction
        if MW_EBV is not None and MW_EBV != 0. and MW_RV is not None:
            self.spectrum.deredden(E_BV=MW_EBV, R_V=MW_RV)

        # Host extinction
        if host_EBV is not None and host_EBV != 0. and host_RV is not None:
            self.spectrum.deredden(E_BV=host_EBV, R_V=host_RV)

        self.spectrum.normalize_flux(method='max')

    def _downsample(self, downsampling):
        """Handle downsampling."""
        n_points = len(self.spectrum)
        sample_limit = 4000

        if int(n_points / downsampling) > sample_limit:
            downsampling = n_points / sample_limit
            msg = (
                f'Flux array is too large (>{sample_limit}). '
                f'Downsampling will be adjusted. '
            )
            self._logger.warning(msg)

        if downsampling <= 1.:
            return

        self.spectrum.downsample(factor=downsampling)
        self.spectrum.normalize_flux(method='max')

        msg = (
            f'Downsampled from {n_points} to {len(self.spectrum)} points '
            f'with a factor of {downsampling:.2f}.\n'
        )
        self._logger.info(msg)

    def _setup_plot(self):
        """Setup the spectrum plot."""
        if self._fig is not None:
            return

        self._fig, self._ax = plt.subplots()

        self._ax.set_xlabel(r'$\mathrm{Rest\ wavelength}\ (\AA)$', size=14)
        self._ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', size=14)
        self._ax.set_ylim(0, 1)

        wave = self.spectrum.wave
        flux = self.spectrum.flux
        error = self.spectrum.error

        self._ax.plot(
            wave, flux, color='k', alpha=0.7, lw=1, zorder=0
        )
        self._ax.fill_between(
            wave, flux - error, flux + error,
            color='grey', alpha=0.5, zorder=-1
        )
