from spextractor import Spextractor


fn = './data/sn2006mo-20061113.21-fast.flm'
spex_args = {
    'z': 0.0459,
    'phot_file': './data/ASAS14ad.dat',
    'time': 56705.5,
    'verbose': False,
    'log': False
}

spex = Spextractor(fn, **spex_args)
