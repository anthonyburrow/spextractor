# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:13:56 2015
Automated pEW and velocity extractor example.
@author: Seméli Papadogiannakis
"""
from __future__ import division, print_function

import matplotlib.pyplot as plt
from spextractor import Spextractor


def report(magnitude, error):
    for k in sorted(magnitude):
        print(k, magnitude[k], '+-', error[k])


fn = 'sn2006mo/sn2006mo-20061113.21-fast.flm'
z = 0.0459
spex = Spextractor(fn, z=z, SNtype='Ia')
spex.process(sigma_outliers=3, downsampling=3, model_uncertainty=True,
             optimize_noise=False, plot=True)

plt.title('Ia_example')
plt.tight_layout()
plt.savefig('Ia_example.png', dpi=300)

plt.close('all')

print('pew (Å)')
report(spex.pew, spex.pew_err)
