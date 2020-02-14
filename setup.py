#!/usr/bin/env python3

import pathlib

import setuptools
from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy as np

basedir = pathlib.Path(__file__).resolve().parent
aboutfile = basedir / 'cm_microtissue_struct' / '__about__.py'
scriptdir = basedir / 'scripts'
readme = basedir / 'README.md'

# Load the info from the about file
about = {}
with aboutfile.open('rt') as fp:
    exec(fp.read(), about)

with readme.open('rt') as fp:
    long_description = fp.read()

scripts = [str(p.relative_to(basedir)) for p in scriptdir.iterdir()
           if not p.name.startswith('.') and p.suffix == '.py' and p.is_file()]

include_dirs = [np.get_include()]

# Cython compile all the things
ext_modules = [
    Extension('cm_microtissue_struct._simulation',
              sources=['cm_microtissue_struct/_simulation.pyx'],
              include_dirs=include_dirs),
]

setuptools.setup(
    name=about['__package_name__'],
    version=about['__version__'],
    url=about['__url__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about['__author__'],
    author_email=about['__author_email__'],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    scripts=scripts,
)
