# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:32:01 2020
Automated pEW and velocity extractor example.
@author: Anthony Burrow
"""

from spextractor import Spextractor
import numpy as np


fn = './spectra/sn2006mo-20061113.21-fast.flm'
z = 0.0459
spex = Spextractor(fn, z=z)

# spex.create_model(downsampling=3)

# Prediction without doing `create_model()` first
# generates a model with default parameters
wave_pred = np.linspace(5500, 6000, 1000)
mean, var = spex.predict(wave_pred)
