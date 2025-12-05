import gpytorch
import numpy as np
from SpectrumCore import Spectrum
import torch

from .InterpolationModel import InterpolationModel


class ExactGPModel(gpytorch.models.ExactGP):
    """GPyTorch GP model with structured kernel interpolation."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        grid_size = gpytorch.utils.grid.choose_grid_size(train_x, 1.0)

        self.mean_module = gpytorch.means.ConstantMean()

        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                base_kernel, grid_size=grid_size, num_dims=1
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # type: ignore[arg-type]


class SKIModel(InterpolationModel):
    """
    Structured Kernel Interpolation GP model using GPyTorch.

    Scales better than exact GP for large datasets using grid-based
    interpolation.

    Usage:
        model = SKIModel(logger)
        model.fit(spectrum)
        y_pred, y_std = model.predict(wavelengths)
    """

    def __init__(self, logger=None):
        """Initialize SKI GP model.

        Parameters
        ----------
        logger : logging.Logger | None, optional
            Logger for diagnostic output. If None, logging is suppressed.
        """
        super().__init__()

        self._logger = logger

        self._model: ExactGPModel | None = None
        self._likelihood: (
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood | None
        ) = None

        self.training_iter = 50
        self._noise_floor = 1e-4

    def fit(self, spectrum: Spectrum) -> ExactGPModel:
        """
        Fit the SKI GP model to a Spectrum object.

        Parameters
        ----------
        spectrum : SpectrumCore.Spectrum
            Spectrum object containing wavelength and flux data.

        Returns
        -------
        ExactGPModel
            Fitted GPyTorch model.
        """
        x = spectrum.wave.astype(np.float32)
        x_norm = self._store_normalization(x)

        train_x = torch.from_numpy(x_norm).float().unsqueeze(-1)
        train_y = torch.from_numpy(spectrum.flux).float()

        if spectrum.has_error:
            noise_vec = torch.from_numpy(
                spectrum.error.astype(np.float32) ** 2
            )
        else:
            # Fallback to homoscedastic noise
            noise_vec = torch.full_like(train_y, 1e-4)

        noise_vec = torch.clamp(noise_vec, min=self._noise_floor)

        self._likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=noise_vec, learn_additional_noise=True
        )
        self._model = ExactGPModel(train_x, train_y, self._likelihood)

        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self._likelihood, self._model
        )

        if self._logger:
            self._logger.info('Created SKI GP model')
            self._logger.info(f'Training iterations: {self.training_iter}')

        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)  # type: ignore[operator]
            loss.backward()
            optimizer.step()

            if self._logger and (i + 1) % 10 == 0:
                self._logger.info(
                    f'Iter {i + 1}/{self.training_iter} - '
                    f'Loss: {loss.item():.3f}'
                )

        if self._logger:
            base_k = self._model.covar_module.base_kernel.base_kernel
            lengthscale = base_k.lengthscale.item()  # type: ignore[union-attr]
            self._logger.info(f'Lengthscale: {lengthscale:.2f}')

        return self._model

    def predict(self, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict flux and uncertainty for given wavelengths.

        Parameters
        ----------
        X_pred : np.ndarray
            Wavelength values to predict at.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Predicted mean and standard deviation.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError('Model must be fit before prediction.')

        self._model.eval()
        self._likelihood.eval()

        x_norm = self._normalize_x(X_pred.astype(np.float32))
        x_test = torch.from_numpy(x_norm).float().unsqueeze(-1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_noise = torch.full((x_test.size(0),), self._noise_floor)
            observed_pred = self._likelihood(self._model(x_test), test_noise)
            mean = observed_pred.mean.numpy()
            std = observed_pred.stddev.numpy()

        return mean, std
