# -*- coding: UTF-8 -*-
"""
Script for training a model.
"""
import os

import pickle

import logging

from sklearn.model_selection import train_test_split

from chicksexer import PACKAGE_ROOT
from chicksexer.model import CharLSTM
from chicksexer.util import set_default_log_level, set_log_level, get_logger, set_default_log_path, \
    set_log_path

_LOG_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'logs')
_MODEL_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'models')
_DATA_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'data')
_TRAIN_DATA_PATH = os.path.join(_DATA_ROOT, 'name2prob.pkl')

# Training constants
_RANDOM_STATE = 0  # this is to make train/test split always return the same split
_VALID_SIZE = 0.1

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def main():
    # config
    model_file_name = 'embsize-128_rnnsize-256_batch-2_lr-0.001'
    log_path = os.path.join(_LOG_ROOT, '{}.log'.format(model_file_name))
    model_path = os.path.join(_MODEL_ROOT, model_file_name)

    # set default log level
    set_default_log_level(logging.DEBUG)
    set_log_level(_LOGGER, logging.DEBUG)

    # write to a log file
    set_default_log_path(log_path)
    set_log_path(_LOGGER, log_path)

    with open(_TRAIN_DATA_PATH, 'rb') as pickle_file:
        name2proba = pickle.load(pickle_file)

    names = list()
    y = list()
    for name, proba in name2proba.items():
        names.append(name)
        y.append(proba)

    # split into train/valid set
    names_train, names_valid, y_train, y_valid = train_test_split(
            names, y, random_state=_RANDOM_STATE, test_size=_VALID_SIZE)

    _LOGGER.info('Initialize CharLSTM object...')
    model = CharLSTM(embedding_size=128, rnn_size=256, learning_rate=0.001, num_rnn_layers=2)
    model.train(names_train, y_train, names_valid, y_valid, model_path, patience=30000000)

if __name__ == '__main__':
    main()
