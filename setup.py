# -*- coding: utf-8 -*-
import setuptools
from setuptools import setup

setup(name='spextractor', version='0.1',
      description='Automatic extractor of spectral features through Gaussian Processes',
      url='https://github.com/anthonyburrow/spextractor',
      author='Anthony Burrow',
      author_email='anthony.r.burrow-1@ou.edu',
      license='GPL-v3',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      optional=['matplotlib'],
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Intended Audience :: Science/Research',
                   'Development Status :: 3 - Alpha']
      )
