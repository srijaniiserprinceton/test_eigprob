#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
time_eigprob - a Python package for timing
eigenvalue problem solvers across
CPU, GPU using numpy, scipy, tensorflow etc
:copyright:
    Srijan B. Das (sbdas@princeton.edu), 2021 
    Samarth G. Kashyap (g.samarth@tifr.res.in), 2021
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
'''
# Importing setuptools monkeypatches some of distutils commands so things like
# 'python setup.py develop' work. Wrap in try/except so it is not an actual
# dependency. Inplace installation with pip works also without importing
# setuptools.

import os
import sys
import math
import argparse
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as TestCommand


setup(
    name='timeeigprob',
    version='0.1.',
    packages=find_packages("."), # Finds every folder with __init__.py in it. (Hehe)
)
