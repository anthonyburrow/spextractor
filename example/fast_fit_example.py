import numpy as np

from spextractor import Spextractor

fn = './data/SN2006mo.dat'
z = 0.0459

params = {
    'plot': True,
    'log': False,
    'z': z,
}
spex = Spextractor(fn, **params)

"""
Prediction without doing `create_model()` first generates a model with default
parameters
"""
spex.create_model(model_type='spline')

wave_pred = np.linspace(5500.0, 6000.0, 1000)
mean = spex.predict(wave_pred)
