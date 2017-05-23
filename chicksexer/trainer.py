# -*- coding: UTF-8 -*-
"""
Script for training a model.
"""
import os

import pickle

import logging

from chicksexer import PACKAGE_ROOT
from chicksexer.model import CharLSTM
from chicksexer.util import set_default_log_level, set_log_level, get_logger, set_default_log_path, \
    set_log_path

_LOG_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir)
_MODEL_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'models')
_DATA_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'data')
_TRAIN_DATA_PATH = os.path.join(_DATA_ROOT, 'name2prob.pkl')

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def main():
    # config
    model_file_name = 'test'
    log_path = os.path.join(_LOG_ROOT, 'model.log')

    # set default log level
    set_default_log_level(logging.DEBUG)
    set_log_level(_LOGGER, logging.DEBUG)

    # write to a log file
    set_default_log_path(log_path)
    set_log_path(_LOGGER, log_path)

    with open(_TRAIN_DATA_PATH, 'rb') as pickle_file:
        name2proba = pickle.load(pickle_file)

    samples = list()
    y = list()
    for name, proba in name2proba.items():
        samples.append(name)
        y.append(proba)

    _LOGGER.info('Initialize CharLSTM object...')
    model = CharLSTM()
    model.train(samples, y, os.path.join(_MODEL_ROOT, model_file_name), profile=True)

if __name__ == '__main__':
    main()
