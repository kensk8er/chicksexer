# -*- coding: UTF-8 -*-
"""
Script for training a model.
"""
import json
import os
import pickle
import logging
from collections import OrderedDict
from random import choice

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from chicksexer.util import set_default_log_level, set_log_level, get_logger, \
    set_default_log_path, set_log_path

_LOG_PATH = os.path.join('logs', 'trainer.log')
_MODEL_ROOT = 'models'
_TRAIN_DATA_PATH = os.path.join('data', 'name2proba.pkl')

# Training constants
_RANDOM_STATE = 0  # this is to make train/test split always return the same split
_VALID_SIZE = 0.1

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def _get_parameter_space():
    """Define the parameter space to explore here."""
    parameter_space = OrderedDict()
    parameter_space.update({'embedding_size': [16 * i for i in range(1, 11)]})
    parameter_space.update({'rnn_size': [32 * i for i in range(1, 31)]})
    parameter_space.update({'num_rnn_layers': [1, 2, 3]})
    parameter_space.update({'learning_rate': [0.0001 * (2 ** i) for i in range(11)]})
    parameter_space.update({'rnn_dropouts': [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]})
    parameter_space.update({'final_dropout': [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]})
    return parameter_space


def main():
    # set default log level
    set_default_log_level(logging.DEBUG)
    set_log_level(_LOGGER, logging.DEBUG)

    # write to a log file
    set_default_log_path(_LOG_PATH)
    set_log_path(_LOGGER, _LOG_PATH)

    names, y = _get_training_data()

    # split into train/valid set
    names_train, names_valid, y_train, y_valid = train_test_split(
            names, y, random_state=_RANDOM_STATE, test_size=_VALID_SIZE)

    _random_search(names_train, names_valid, y_train, y_valid, _get_parameter_space())


def _get_training_data():
    names = list()
    y = list()

    with open(_TRAIN_DATA_PATH, 'rb') as pickle_file:
        name2proba = pickle.load(pickle_file)

    for name, proba in name2proba.items():
        names.append(name)
        y.append(proba)

    return names, y


def _random_search(names_train, names_valid, y_train, y_valid, parameter_space):
    """Perform random search over given hyper-parameter space."""

    def sample_parameters(parameter_space):
        parameters = OrderedDict()
        for key, vals in parameter_space.items():
            if key == 'rnn_dropouts':
                sampled_val = list()
                for _ in range(parameters['num_rnn_layers']):
                    sampled_val.append(choice(vals))  # sample a value randomly
            else:
                sampled_val = choice(vals)  # sample a value randomly
            parameters[key] = sampled_val
        return parameters

    def construct_model_name(parameters):
        def format_precision(number):
            if isinstance(number, float):
                return '{:.5f}'.format(number).rstrip('0')
            else:
                return str(number)

        def format_val(val):
            if isinstance(val, list):
                return '-'.join(format_precision(ele) for ele in val)
            else:
                return format_precision(val)

        return '_'.join('{}-{}'.format(key, format_val(val)) for key, val in parameters.items())

    from chicksexer.classifier import CharLSTM  # import here after you configure logging
    searched_parameters = set()
    best_valid_score = np.float64('-inf')
    best_parameters = None
    count = 1

    try:
        while True:
            parameters = sample_parameters(parameter_space)
            if str(parameters) in searched_parameters:
                continue

            _LOGGER.info('---------- ({}) Start experimenting with a new parameter set ----------\n'
                         .format(count))
            _LOGGER.info('Hyper-parameters:\n{}'.format(json.dumps(parameters, indent=2)))

            # construct the model name
            model_name = construct_model_name(parameters)
            model_path = os.path.join(_MODEL_ROOT, model_name)
            _LOGGER.info('Model name: {}'.format(model_name))

            _LOGGER.info('Initialize CharLSTM object with the new parameters...')
            model = CharLSTM(**parameters)

            _LOGGER.info('Started the train() method...')
            score = model.train(names_train, y_train, names_valid, y_valid, model_path)
            searched_parameters.add(str(parameters))

            if score > best_valid_score:
                _LOGGER.info('Achieved best validation score so far in the search.')
                _LOGGER.info('Hyper-parameters:\n{}'.format(json.dumps(parameters, indent=2)))
                best_valid_score = score
                best_parameters = parameters

            _LOGGER.info('-------- ({}) Finished experimenting with the parameter set --------\n\n'
                         .format(count))
            count += 1

    except KeyboardInterrupt:
        _LOGGER.info('Random Search finishes because of Keyboard Interrupt.')
        _LOGGER.info('Best Validation Score: {:.3f}'.format(best_valid_score))
        _LOGGER.info('Best Hyper-parameters:\n{}'.format(json.dumps(best_parameters, indent=2)))

    except InvalidArgumentError as error:
        _LOGGER.exception(error)
        _LOGGER.info('-------- ({}) Skip the parameter set --------\n\n'.format(count))


if __name__ == '__main__':
    main()
