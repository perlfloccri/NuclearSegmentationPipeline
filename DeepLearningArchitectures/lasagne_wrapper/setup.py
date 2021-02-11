##!/usr/bin/env python   
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          setup.py
# Purpose:       install
#
# Authors:       Matthias Dorfer
#
# License:       LGPL or BSD
#-------------------------------------------------------------------------------

#import os
from setuptools import setup, find_packages

from lasagne_wrapper import __version__
version = __version__ # @UndefinedVariable

DESCRIPTION = 'A Toolkit for training and applying lasagne networks.'
DESCRIPTION_LONG = """A Toolkit for training and applying lasagne networks."""

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Environment :: Web Environment',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Artistic Software',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

if __name__ == '__main__':
    setup(
        name='lasagne_wrapper',
        version=version,
        description=DESCRIPTION,
        long_description=DESCRIPTION_LONG,
        author='Matthias Dorfer',
        author_email='matthias.dorfer@jku.at',
        license='BSD',
        packages=find_packages(exclude=['ez_setup']),
        include_package_data=True,
    )

#------------------------------------------------------------------------------
# eof
