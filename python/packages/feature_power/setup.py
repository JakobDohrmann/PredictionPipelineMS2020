#! /usr/bin/env python
"""Aggregate Feature Power for GMBs."""

import codecs
import os

from setuptools import find_packages, setup

DISTNAME = 'feature_power'
DESCRIPTION = 'Aggregate Feature Power for GMBs.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Jakob Dohrmann'
MAINTAINER_EMAIL = 'jakobd@mail.sfsu.edu'
URL = 'https://github.com/JakobDohrmann/PredictionPipelineMS2020'
LICENSE = 'GPLv3'
DOWNLOAD_URL = 'https://github.com/JakobDohrmann/PredictionPipelineMS2020'
VERSION ="0.0.1"
INSTALL_REQUIRES = ['gmpy']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3']
EXTRAS_REQUIRE = {
    'tests': [
        'pandas',
        'lighgbm'],
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
