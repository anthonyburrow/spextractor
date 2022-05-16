import numpy as np
import GPy


def model(data, optimize_noise=False, logger=None):
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
    x = data[:, 0]
    y = data[:, 1]
    y_err = data[:, 2]

    kernel = GPy.kern.Matern32(1, lengthscale=300., variance=0.001)

    model_uncertainty = False
    if np.any(y_err):
        model_uncertainty = True
    else:
        optimize_noise = True
        msg = ('No flux uncertainty detected - optimizing noise parameter.')
        logger.info(msg)

    # Add flux errors as noise to kernel
    kern = kernel
    if model_uncertainty:
        diag_vars = y_err**2 * np.eye(len(y_err))
        kern_uncertainty = GPy.kern.Fixed(1, diag_vars)
        kern = kernel + kern_uncertainty
        logger.info('Flux error added to GPy kernel')

    # Create model
    m = GPy.models.GPRegression(x[:, np.newaxis], y[:, np.newaxis], kern)
    m['Gaussian.noise.variance'][0] = 0.01

    logger.info('Created GP')

    # Optimize model
    if model_uncertainty:
        m['.*fixed.variance'].constrain_fixed()

    if not optimize_noise:
        m.Gaussian_noise.fix(1e-6)

    m.optimize(optimizer='bfgs')

    logger.info(m)

    if model_uncertainty:
        # Use optimized hyperparameters with original kernel
        kernel.lengthscale = kern.Mat32.lengthscale
        kernel.variance = kern.Mat32.variance

    return m, kernel


def predict(x_pred, model, kernel):
    mean, var = model.predict(x_pred[:, np.newaxis], kern=kernel.copy())

    return mean.squeeze(), var.squeeze()
