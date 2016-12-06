try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Distutils import Extension, build_ext
import numpy as np
import sys
import glob
import os

include_dirs = [np.get_include()]

args = sys.argv[1:]

# get rid of intermediate and library files
if "clean" in args:
    print("Deleting cython files...")
    to_remove = []
    # C
    to_remove += glob.glob('bitpacking/packing.c')
    to_remove += glob.glob('bitpacking/bitcount.c')
    # Static lib files
    to_remove += glob.glob('bitpacking/*.so')
    for f in to_remove:
        os.remove(f)


# We want to always use build_ext --inplace
if args.count('build_ext') > 0 and args.count('--inplace') == 0:
    sys.argv.insert(sys.argv.index('build_ext')+1, '--inplace')

extensions = [
    Extension('bitpacking.packing',
              ['bitpacking/packing.pyx'],
              include_dirs=include_dirs),
    Extension('bitpacking.bitcount',
              ['bitpacking/bitcount.pyx'],
              libraries=['gmp'],
              include_dirs=include_dirs),
    ]

setup(
    name='bitpacking',
    include_dirs=[np.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions
    )
