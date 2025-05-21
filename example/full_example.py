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

- Data is provided here in the form of a file path (str) or data array
  (numpy.ndarray).
- Redshift specified here.
- Other parameters available (see constructor docstring).
'''
spex = Spextractor(fn, z=z, plot=True)

'''
Make specifications to the model and any downsampling that occurs beforehand.

- This includes sigma-clipping, downsampling, etc. parameters (see docstring).
'''
spex.create_model(downsampling=3.)

'''
Do feature calculations (velocity, pEW, depth, etc.).

- Features are based on those provided in "spextractor/physics/lines.py".
- Can specify only specific features to process with `features` argument.
- Features may be retrieved through the dictionary attributes `.pew`,
  `.pew_err`, `.vel`, `.vel_err`, etc.
- Keys to dictionary attributes are the strings again found in the "lines.py",
  such that each feature has an associated velocity, pEW, etc..
'''
si = 'Si II 6150A'
spex.process()

vsi = spex.vel[si]
vsi_err = spex.vel_err[si]
print(f'vsi = {vsi:.3f} +- {vsi_err:.3f}')

'''
Retrieve the spectrum plot if it was created.

- The `.plot` attribute returns a tuple of a pyplot figure and axis of the
  plot.
'''
fig, ax = spex.plot

ax.set_title('Ia_example')

plt.tight_layout()
fig.savefig('Ia_example.png', dpi=300)

'''
May want to close the plots if running on a large number of spectra to prevent
huge memory leaks. (This should always be dealt with when using matplotlib
like this.)
'''
plt.close('all')
