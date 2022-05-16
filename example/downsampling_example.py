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

'''
Downsample the spectrum with the constraint that photon flux is conserved in
each bin.
'''
downsample_factor = 3.
ds_data = downsample(data, binning=downsample_factor)

print(data.shape)
print(ds_data.shape)
print(len(data) / len(ds_data))
