# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:24:54 2020
Automated pEW and velocity extractor example.
@author: Anthony Burrow
"""

from spextractor.physics.downsample import downsample
import numpy as np


fn = './spectra/sn2006mo-20061113.21-fast.flm'
data = np.loadtxt(fn)
wave = data[:, 0]
flux = data[:, 1]
flux_err = data[:, 1]

'''
Downsample the spectrum with the constraint that photon flux is conserved in
each bin.
'''
downsample_factor = 3.
ds_wave, ds_flux, ds_flux_err = \
    downsample(wave, flux, flux_err, binning=downsample_factor)

print(flux.shape)
print(ds_flux.shape)
print(len(flux) / len(ds_flux))
