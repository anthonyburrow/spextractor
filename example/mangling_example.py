from spextractor import Spextractor


# ASAS14ad.dat is not included so this will fail; but this is how this is done.
# The photometry file needs to be in SNooPy-readable format

fn = './data/sn2006mo-20061113.21-fast.flm'
spex_args = {
    'z': 0.0459,
    'phot_file': './data/ASAS14ad.dat',
    'time': 56705.5,
    'verbose': False,
    'log': False
}

spex = Spextractor(fn, **spex_args)
