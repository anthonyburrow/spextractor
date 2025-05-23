# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:16:35 2020
Automated pEW and velocity extractor example.
@author: Anthony Burrow
"""

import matplotlib.pyplot as plt
from spextractor import Spextractor


fn = './data/SN2006mo.dat'
z = 0.0459

'''
Create the Spextractor instance.

- For specifying a manual feature range for vel/pEW calculations, set
  `manual_range` to True here.
- This selection process occurs during this instantiation step.
- Close the plot when finished with selection.
'''

spex = Spextractor(fn, z=z, plot=True)

'''
Continue as normal from here with the updated feature ranges.
'''
spex.create_model(downsampling=3.)

spex.process(manual_range=True)

si = 'Si II 6150A'
vsi = spex.vel[si]
vsi_err = spex.vel_err[si]
print(f'vsi = {vsi:.3f} +- {vsi_err:.3f}')

fig, ax = spex.plot

plt.tight_layout()
fig.savefig('Ia_example.png', dpi=300)

plt.close('all')

print(spex.vel)
