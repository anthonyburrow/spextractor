# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:35:00 2024
Automated pEW and velocity extractor example.
@author: Anthony Burrow
"""

import matplotlib.pyplot as plt
from spextractor import Spextractor


fn = './data/sn2006mo-20061113.21-fast.flm'
z = 0.0459


spex = Spextractor(fn, z=z, plot=True)
spex.create_model(downsampling=3.)

'''
Calculate the blue-edge velocity with `velocity_method` set to 'blue_edge'.
'''
params = {
    'features': ('Si II 5800A', 'Si II 6150A'),
    'velocity_method': 'blue_edge',
}
spex.process(**params)


si = 'Si II 6150A'

vsi = spex.vel[si]
vsi_err = spex.vel_err[si]
print(f'vsi = {vsi:.3f} +- {vsi_err:.3f}')


fig, ax = spex.plot

ax.set_title('Ia_example')

plt.tight_layout()
fig.savefig('Ia_example.png', dpi=300)

plt.close('all')
