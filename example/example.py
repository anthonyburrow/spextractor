# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:16:35 2020
Automated pEW and velocity extractor example.
@author: Anthony Burrow
"""

import matplotlib.pyplot as plt
from spextractor import Spextractor


def report(magnitude, error):
    for k in sorted(magnitude):
        print(k, magnitude[k], '+-', error[k])


fn = './spectra/sn2006mo-20061113.21-fast.flm'
z = 0.0459
spex = Spextractor(fn, z=z)
spex.create_model(sigma_outliers=3, downsampling=3, model_uncertainty=True,
                  optimize_noise=False)
spex.process(plot=True)

fig, ax = spex.plot

ax.set_title('Ia_example')

plt.tight_layout()
plt.savefig('Ia_example.png', dpi=300)

plt.close('all')

print('pew (Å)')
report(spex.pew, spex.pew_err)
