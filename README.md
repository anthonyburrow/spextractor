# spextractor

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/anthonyburrow/spextractor/run_pytest.yml)](https://github.com/anthonyburrow/spextractor/actions/workflows/run_pytest.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Fork of the original spextractor (<https://github.com/astrobarn/spextractor>).

**spextractor** is a Python package designed to extract line velocities and
pseudo-equivalent widths (pEWs) from astronomical spectra using Gaussian
process regression. While originally tailored for Type Ia supernovae (SNe Ia),
this code is adaptable to other spectral types and features.

---

## Features

- **Gaussian Process Smoothing**: Applies GPR to model spectral data, providing smooth interpolations and uncertainty estimates.
- **Feature Extraction**: Calculates velocities and pEWs for specified spectral features.
- **Uncertainty Estimation**: Offers statistical error estimates for measurements, with recent updates replacing deprecated methods with more robust alternatives.
- **Extensibility**: Easily adaptable to different spectral features and types beyond SNe Ia.

---

## Installation

Clone the repository and install the package using `pip`:

```bash
git clone https://github.com/anthonyburrow/spextractor.git
cd spextractor
pip install .
```

---

## Usage

See the examples found in the `examples` directory.

---

## Acknowledgements

This repository is a fork of the original
[spextractor](https://github.com/astrobarn/spextractor) by
[Seméli Papadogiannakis](https://github.com/astrobarn).
