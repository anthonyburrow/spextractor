# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:32:01 2020
Quick-fitting/smoothing example.
@author: Anthony Burrow
"""

from spextractor import Spextractor
import numpy as np


fn = './spectra/sn2006mo-20061113.21-fast.flm'
z = 0.0459

spex = Spextractor(fn, z=z, verbose=False, log=True)

'''
Prediction without doing `create_model()` first generates a model with default
parameters
'''
spex.create_model(downsampling=3.)

wave_pred = np.linspace(5500., 6000., 1000)
mean, var = spex.predict(wave_pred)
