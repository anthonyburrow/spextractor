import numpy as np
import matplotlib.pyplot as plt

from SpectrumCore.Spectrum import Spectrum
from SpectrumCore.util.interpolate import interp_linear

from .util.log import setup_log
from .util.manual import ManualRange
from .physics.feature import Feature
from .physics.lines import sn_types, sn_lines
from .math import gpr


class Spextractor:

    def __init__(
        self, data, wave_unit=None, plot=False, log=False, *args, **kwargs
    ):
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

        # Undefined instance attributes
        self._model = None

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

        model = gpr.model(self.spectrum, logger=self._logger)

        self._model = model

        return model

    def reset_model(self):
        """Clears the current GPR model."""
        self._model = None

    def predict(self, X_pred):
        """Use the created GPR model to make a prediction at any given points.

        Parameters
        ----------
        X_pred : numpy.ndarray
            Test input set at which to predict.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            A tuple consisting of the mean and uncertainty values calculated at
            each of the prediction points.
        """
        return gpr.predict(X_pred, self.model)

    def process(
            self, features: tuple[str] = None, predict_res: int = 2000,
            sn_type: str = None, manual_range: bool = False, *args, **kwargs
    ):
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
        gpr_wave = np.linspace(
            self.spectrum.wave_start, self.spectrum.wave_end, predict_res
        )
        gpr_mean, gpr_std = gpr.predict(gpr_wave, self.model)
        gpr_spectrum = Spectrum(np.c_[gpr_wave, gpr_mean, gpr_std])

        if features is None:
            if sn_type is None:
                sn_type = 'Ia'
            features = sn_types[sn_type]

        feature_list = []
        for feature in features:
            feature_info = sn_lines[feature]

            lo_range = feature_info['lo_range']
            hi_range = feature_info['hi_range']

            if lo_range[0] < gpr_spectrum.wave_start:
                continue
            if gpr_spectrum.wave_end < hi_range[1]:
                continue

            feature_obj = Feature(
                name=feature,
                rest_wave=feature_info['rest'],
                spectrum=gpr_spectrum,
                gpr_model=self.model,
            )

            feature_obj.update_endpoints(lo_range, hi_range)

            feature_list.append(feature_obj)

        if manual_range:
            self._logger.info('Manually changing feature bounds...')
            if self._fig is not None:
                self._fig, self._ax = None
                plt.close('all')

            _ = ManualRange(self.spectrum, feature_list, self._logger)

        if self._plot and self._fig is None:
            self._setup_plot()

        for feature in feature_list:
            vel, vel_err, draw_point = feature.velocity(*args, **kwargs)

            self.vel[feature.name] = vel
            self.vel_err[feature.name] = vel_err

            if self._plot:
                self._ax.axvline(
                    draw_point[0], ymax=draw_point[1],
                    color='k', linestyle='--'
                )
                self._ax.text(
                    draw_point[0] + 30., 0.015, feature.name,
                    rotation=90., fontsize=8.
                )

            pew, pew_err = feature.pEW(*args, **kwargs)
            self.pew[feature.name] = pew
            self.pew_err[feature.name] = pew_err

            if self._plot:
                data = feature.feature_data
                feat_range = data[[0, -1], :2]
                continuum = interp_linear(data[:, 0], feat_range)

                self._ax.scatter(
                    feat_range[:, 0], feat_range[:, 1], color='k', s=30
                )
                self._ax.fill_between(
                    data[:, 0], data[:, 1], continuum,
                    color='#00a3cc', alpha=0.3
                )

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
        if self._plot and self._fig is None:
            self._setup_plot()
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
        self._fig, self._ax = plt.subplots()

        self._ax.set_xlabel(r'$\mathrm{Rest\ wavelength}\ (\AA)$', size=14)
        self._ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', size=14)
        self._ax.set_ylim(0., 1.)

        wave = self.spectrum.wave
        flux = self.spectrum.flux

        # Display (preprocessed) original data
        self._ax.plot(
            wave, flux, color='k', alpha=0.7, lw=1, zorder=0
        )
        if self.spectrum.has_error:
            error = self.spectrum.error
            self._ax.fill_between(
                wave, flux - error, flux + error,
                color='grey', alpha=0.5, zorder=-1
            )

        # Display the GPR
        wave_pred = np.linspace(
            self.spectrum.wave_start, self.spectrum.wave_end, 2000
        )
        mean, std = gpr.predict(wave_pred, self.model)

        self._ax.plot(wave_pred, mean, color='red', zorder=2, lw=1)
        self._ax.fill_between(
            wave_pred, mean - std, mean + std,
            alpha=0.3, color='red', zorder=1
        )
