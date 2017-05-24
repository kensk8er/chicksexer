# -*- coding: UTF-8 -*-
"""
chicksexer package
"""
import os

from ._version import __version__

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))

__author__ = 'kensk8er'

# define modules/functions to expose under here (they might need PACKAGE_ROOT)
from .api import predict_genders, predict_gender
