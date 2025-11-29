import numpy as np


def gaussian(
    X: float | np.ndarray, mu: float, sigma: float, A: float, const: float
) -> float | np.ndarray:
    """Unnormalized Gaussian function."""
    return A * np.exp(-0.5 * ((X - mu) / sigma) ** 2) + const
