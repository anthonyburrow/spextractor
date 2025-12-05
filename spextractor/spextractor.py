from collections.abc import Iterable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from SpectrumCore.plot import basic_spectrum
from SpectrumCore.Spectrum import Spectrum
from SpectrumCore.util.interpolate import interp_linear

from .math.GaussianProcessModel import GaussianProcessModel
from .math.InterpolationModel import InterpolationModel
from .math.SplineModel import SplineModel

try:
    from .math.SKIModel import SKIModel

    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

from .physics.feature import Feature
from .physics.lines import FEATURE_RANGES, FEATURE_REST_WAVES, sn_types
from .util.log import setup_log
from .util.manual import ManualRange


class Spextractor:
    def __init__(
        self,
        data: str | np.ndarray,
        plot: bool = False,
        log: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for the Spextractor class.

        Parameters
        ----------
        data : str, numpy.ndarray
            Spectral data (can be unnormalized) where columns are wavelengths
            (Angstroms), flux, and flux uncertainty.
        plot : bool, optional
            Create and hold a plot of the data, interpolation model
            prediction, and velocity/pEW information that may be
            calculated. False by default.
        log : bool, optional
            Determines whether or not a file is created to log console output.

        **kwargs
            wave_unit : str, optional
                Unit of wavelength for the input data. Available units are
                'angstrom' and 'micron'. If None, this defaults to 'angstrom'.
            remove_telluric: bool, optional
                Remove telluric features. Default is False.
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
        self._logger = setup_log(
            *args, filename=log_fn, log_to_file=log, **kwargs
        )

        self.spectrum = Spectrum(data, *args, **kwargs)
        self._preprocess_spectrum(*args, **kwargs)

        self._plot = plot
        self._fig: Figure | None = None
        self._ax: Axes | None = None

        # Undefined instance attributes
        self._model: InterpolationModel | None = None

        self.pew = {}
        self.pew_err = {}
        self.vel = {}
        self.vel_err = {}
        self.depth = {}
        self.depth_err = {}

    def create_model(
        self,
        model_type: str = 'gpr',
        downsampling: float | None = None,
        *args,
        **kwargs,
    ) -> InterpolationModel:
        """Makes specifications to the interpolation model.

        Parameters
        ----------
        model_type : str, optional
            Type of model to create. Options are 'gpr' for Gaussian Process
            Regression, 'spline' for UnivariateSpline, or 'ski' for
            Structured Kernel Interpolation GP. Default is 'gpr'.
        downsampling : float, optional
            Downsampling factor used for the model fit. Downsampled data will
            thus be `1 / downsampling` of its original size. If `downsampling
            <= 1.`, the original data will be used.

        Returns
        -------
        InterpolationModel
            The interpolation model that is created.
        """
        downsampling = downsampling or 1.0
        self._downsample(downsampling)

        model_type = model_type.lower() or 'gpr'
        if model_type == 'gpr':
            model = GaussianProcessModel(self._logger)
        elif model_type == 'spline':
            model = SplineModel(self._logger)
        elif model_type == 'ski':
            if not HAS_GPYTORCH:
                raise ImportError(
                    'SKI model requires GPyTorch. '
                    'Install with: `make install-gpytorch-gpu` or '
                    '`make install-gpytorch-cpu`'
                )
            model = SKIModel(self._logger)
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose 'gpr', 'spline', or 'ski'."
            )

        model.fit(self.spectrum)

        self._model = model
        return model

    def reset_model(self) -> None:
        """Clears the current interpolation model."""
        self._model = None

    def predict(
        self, X_pred: np.ndarray
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Use the created interpolation model to predict at given points.

        Parameters
        ----------
        X_pred : numpy.ndarray
            Test input set at which to predict.

        Returns
        -------
        numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
            For Gaussian Process model: a tuple of (mean, uncertainty).
            For spline model: array of predicted mean values only.
        """
        wave_factor = self.spectrum.wave_factor
        return self.model.predict(X_pred * wave_factor)

    def process(
        self,
        features: Iterable[str] | None = None,
        predict_res: int = 2000,
        sn_type: str | None = None,
        manual_range: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Calculate the line velocities, pEWs, and line depths of each
           feature.

        Parameters
        ----------
        features : Iterable[str], optional
            Iterable (list, tuple, etc.) containing strings of features for
            which to calculate properties. These must be included in
            "./physics/lines.py". By default, every feature in "lines.py" is
            processed.
        predict_res : int, optional
            Sample size (resolution) of values predicted by interpolation
            model.
        sn_type : str, optional
            Type of SN, used to determine which features are able to be
            processed. This must be a key in "./physics/lines.py" ('Ia', 'Ib',
            or 'Ic'). If None, 'Ia' is used by default.
        manual_range : bool, optional
            Used for manually setting feature minimum/maximum ranges via a
            spectrum plot. False by default.

        **kwargs:
            velocity_method
        """
        model_wave = np.linspace(
            self.spectrum.wave_start, self.spectrum.wave_end, predict_res
        )
        result = self.model.predict(model_wave)

        if isinstance(result, tuple):
            model_mean, model_std = result
            model_spectrum = Spectrum(np.c_[model_wave, model_mean, model_std])
        else:
            model_mean = result
            model_spectrum = Spectrum(np.c_[model_wave, model_mean])

        if features is None:
            if sn_type is None:
                sn_type = 'Ia'
            features = sn_types[sn_type]

        feature_list = []
        for feature in features:
            rest_wave = FEATURE_REST_WAVES[feature]
            ranges = FEATURE_RANGES[feature]

            lo_range = ranges['lo_range']
            hi_range = ranges['hi_range']

            if lo_range[0] < model_spectrum.wave_start:
                continue
            if model_spectrum.wave_end < hi_range[1]:
                continue

            feature_obj = Feature(
                name=feature,
                rest_wave=rest_wave,
                spectrum=model_spectrum,
                model=self.model,
            )

            feature_obj.update_endpoints(lo_range, hi_range)

            feature_list.append(feature_obj)

        if manual_range:
            self._logger.info('Manually changing feature bounds...')
            if self._fig is not None:
                self._fig, self._ax = None, None
                plt.close('all')

            _ = ManualRange(self.spectrum, feature_list, self._logger)

        if self._plot and self._fig is None:
            self._setup_plot()

        for feature in feature_list:
            vel, vel_err, draw_point = feature.velocity(*args, **kwargs)

            self.vel[feature.name] = vel
            self.vel_err[feature.name] = vel_err

            if np.isnan(vel) or vel < 0.0:
                self.pew[feature.name] = np.nan
                self.pew_err[feature.name] = np.nan
                continue

            self.vel[feature.name] = vel
            self.vel_err[feature.name] = vel_err

            wave_factor = self.spectrum.wave_factor
            if self._plot and self._ax is not None:
                self._ax.axvline(
                    draw_point[0] / wave_factor,
                    ymax=draw_point[1],
                    color='k',
                    linestyle='--',
                )
                self._ax.text(
                    (draw_point[0] + 30.0) / wave_factor,
                    0.015,
                    feature.name,
                    rotation=90.0,
                    fontsize=8.0,
                )

            pew, pew_err = feature.pEW(*args, **kwargs)

            self.pew[feature.name] = pew
            self.pew_err[feature.name] = pew_err

            if np.isnan(pew) or pew < 0.0:
                continue

            if self._plot and self._ax is not None:
                data = feature.feature_data
                feat_range = data[[0, -1], :2]
                continuum = interp_linear(data[:, 0], feat_range)

                self._ax.scatter(
                    feat_range[:, 0] / wave_factor,
                    feat_range[:, 1],
                    color='k',
                    s=30,
                )
                self._ax.fill_between(
                    data[:, 0] / wave_factor,
                    data[:, 1],
                    continuum,
                    color='#00a3cc',
                    alpha=0.3,
                )

        self._logger.handlers = []  # Close log handlers between instantiations

    @property
    def model(self) -> InterpolationModel:
        if self._model is None:
            msg = (
                'Attempted to use model without generating first. '
                'Creating model with default arguments...'
            )
            self._logger.warning(msg)
            self._model = self.create_model()

        return self._model

    @property
    def rsi(self) -> float:
        try:
            ld5800 = self.depth['Si II 5800A']
            ld6150 = self.depth['Si II 6150A']
            rsi = ld5800 / ld6150
        except KeyError:
            rsi = np.nan

        return rsi

    @property
    def rsi_err(self) -> float:
        try:
            ld5800_err = self.depth_err['Si II 5800A']
            ld6150_err = self.depth_err['Si II 6150A']
            rsi_err = np.sqrt(ld5800_err**2 + ld6150_err**2)
        except KeyError:
            rsi_err = np.nan

        return rsi_err

    @property
    def plot(self) -> tuple[Figure, Axes]:
        if self._fig is None or self._ax is None:
            self._fig, self._ax = self._setup_plot()
        return self._fig, self._ax

    @property
    def wave(self) -> np.ndarray:
        return self.spectrum.wave

    @property
    def flux(self) -> np.ndarray:
        return self.spectrum.flux

    @property
    def flux_denormalized(self) -> np.ndarray:
        return self.spectrum.flux_denormalized

    @property
    def flux_norm(self) -> float:
        return self.spectrum.flux_norm

    def _preprocess_spectrum(
        self,
        z: float | None = None,
        wave_range: tuple[float, float] | None = None,
        remove_telluric=False,
        host_EBV: float | None = None,
        host_RV: float | None = None,
        MW_EBV: float | None = None,
        MW_RV: float = 3.1,
        *args,
        **kwargs,
    ) -> None:
        self.spectrum.remove_nans()
        self.spectrum.remove_nonpositive()

        if remove_telluric:
            self.spectrum.remove_telluric()  # TODO: AFTER MANGLING

        if z is not None and z != 0.0:
            self.spectrum.deredshift(z)

        if wave_range is not None:
            self.spectrum.prune(wave_range)

        # Milky Way extinction
        if MW_EBV is not None and MW_EBV != 0.0 and MW_RV is not None:
            self.spectrum.deredden(E_BV=MW_EBV, R_V=MW_RV)

        # Host extinction
        if host_EBV is not None and host_EBV != 0.0 and host_RV is not None:
            self.spectrum.deredden(E_BV=host_EBV, R_V=host_RV)

        self.spectrum.normalize_flux(method='max')

    def _downsample(self, downsampling: float) -> None:
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

        if downsampling <= 1.0:
            return

        self.spectrum.downsample(factor=downsampling)
        self.spectrum.normalize_flux(method='max')

        msg = (
            f'Downsampled from {n_points} to {len(self.spectrum)} points '
            f'with a factor of {downsampling:.2f}.\n'
        )
        self._logger.info(msg)

    def _setup_plot(self) -> tuple[Figure, Axes]:
        """Setup the spectrum plot."""
        self._fig, self._ax = basic_spectrum()

        wave_factor = self.spectrum.wave_factor
        wave = self.spectrum.wave / wave_factor
        flux = self.spectrum.flux

        if wave_factor == 1.0:
            self._ax.set_xlabel(r'$\mathrm{Rest\ wavelength}\ (\AA)$', size=14)
        elif wave_factor == 1e4:
            self._ax.set_xlabel(
                r'$\mathrm{Rest\ wavelength}\ (\mu m)$', size=14
            )

        self._ax.set_ylabel(r'$\mathrm{Normalized\ flux}$', size=14)
        self._ax.set_ylim(0.0, 1.0)

        # Display (preprocessed) original data
        self._ax.plot(wave, flux, color='k', alpha=0.7, lw=1, zorder=0)
        if self.spectrum.has_error:
            error = self.spectrum.error
            self._ax.fill_between(
                wave,
                flux - error,
                flux + error,
                color='grey',
                alpha=0.5,
                zorder=-1,
            )

        # Display the model prediction
        wave_pred = np.linspace(
            self.spectrum.wave_start, self.spectrum.wave_end, 2000
        )
        result = self.model.predict(wave_pred)
        wave_pred /= wave_factor

        if isinstance(result, tuple):
            mean, std = result
            self._ax.plot(wave_pred, mean, color='red', zorder=2, lw=1)
            self._ax.fill_between(
                wave_pred,
                mean - std,
                mean + std,
                alpha=0.3,
                color='red',
                zorder=1,
            )
        else:
            mean = result
            self._ax.plot(wave_pred, mean, color='red', zorder=2, lw=1)

        return self._fig, self._ax
