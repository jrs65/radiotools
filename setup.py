# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

import radiotools

# Enable OpenMP support if available
if sys.platform == 'darwin':
    compile_args = []
    link_args = []
else:
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

# Cython module for fast operations
proj_ext = Extension(
    "radiotools._proj",
    ["radiotools/_proj.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name='radiotools',
    version=radiotools.__version__,
    license='MIT',

    packages=find_packages(),

    ext_modules=cythonize([proj_ext]),

    install_requires=['Cython>0.18', 'numpy>=1.7', 'scipy>=0.10',
                      'pyfits'],

    author="Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Miscellaneous radio tools for analysis and power spectrum estimation.",
    url="http://github.com/jrs65/radiotools/",
)
