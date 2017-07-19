# -*- coding: UTF-8 -*-
"""
Command Line Interface (CLI) for training a model.

Usage:
    trainer.py random-search [options]
    trainer.py [options]
    trainer.py -h | --help
    trainer.py -v | --version

Commands:
    random-search  Perform a random-search over hyper-parameter space.

Options:
    # universal options for training
    --train-data-path=<str>  Path to the training data file [default: ./data/name2proba_train.pkl]
    --model-dir=<str>  Path to the model directory [default: ./models]
    --batch-size=<int>  The number of samples per batch [default: 128]
    --patience=<int>  The number of iterations to keep training [default: 1024000]
    --valid-size=<float>  The proportion of dataset to use for validation [default: 0.1]
    --profile  Profile the training (profile_train/valid.json will be created)

    # options for when not doing random-search (ignored when doing random-search)
    --embed-size=<int>  The number of dimensions of the character embedding layer [default: 32]
    --char-rnn-size=<int>  The number of dimensions of the character-RNN layer [default: 128]
    --word-rnn-size=<int>  The number of dimensions of the word-RNN layer [default: 128]
    --learning-rate=<float>  Initial learning rate of SGD (Adam Optimizer) [default: 0.001]
    --embed-dropout=<float>  Dropout rate after the embedding layer [default: 0.]
    --char-rnn-dropout=<float>  Dropout rate after the character-RNN layer [default: 0.]
    --word-rnn-dropout=<float>  Dropout rate after the word-RNN layer [default: 0.]

    # universal non-training-related options
    -h --help  Show this screen
    -v --version  Show version
    --verbose  Show debug messages
    --log-path=<str>  Path to the log file [default: ./logs/trainer.log]

Examples:
    python trainer.py random-search
    python trainer.py --embed-size=128 --char-rnn-size=128 --word-rnn-size=128 --char-rnn-dropout=0.5

"""
import json
import os
import pickle
import logging
from collections import OrderedDict
from random import choice

import numpy as np
from docopt import docopt
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from chicksexer.util import set_default_log_level, set_log_level, get_logger, \
    set_default_log_path, set_log_path
from chicksexer import __version__
from chicksexer.classifier import CharLSTM, set_log_path as set_classifier_log_path

_RANDOM_STATE = 0  # this is to make train/test split always return the same split
_HOME_DIR = '~/'

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def _get_parameter_space():
    """Define the parameter space to explore here."""
    parameter_space = OrderedDict()
    parameter_space.update({'embedding_size': [16 * i for i in range(1, 5)]})
    parameter_space.update({'char_rnn_size': [64 * i for i in range(1, 5)]})
    parameter_space.update({'word_rnn_size': [64 * i for i in range(1, 5)]})
    parameter_space.update({'learning_rate': [0.0001 * (2 ** i) for i in range(8)]})
    parameter_space.update({'embedding_dropout': [0., 0.01, 0.03, 0.05, 0.1]})
    parameter_space.update({'char_rnn_dropout': [0., 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]})
    parameter_space.update({'word_rnn_dropout': [0., 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]})
    return parameter_space


def _expand_user_path(args):
    """Expand to absolute path when ~/ appears in the path."""
    for arg_key, arg_val in args.items():
        if isinstance(arg_val, str) and arg_val.startswith(_HOME_DIR):
            args[arg_key] = os.path.expanduser(arg_val)
    return args


def _get_training_data(train_data_path):
    """Load training data from the file and return them."""
    names = list()
    y = list()

    _LOGGER.debug('Loading training data...')
    with open(train_data_path, 'rb') as pickle_file:
        name2proba = pickle.load(pickle_file)

    for name, proba in name2proba.items():
        names.append(name)
        y.append(proba)

    return names, y


def _construct_model_name(parameters):
    """Construct model name from the parameters and return it."""
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

    parameter_names = list(parameters.keys())
    parameter_names.sort()  # sort by key in order to have consistency in the order of params
    return '_'.join('{}-{}'.format(key, format_val(parameters[key])) for key in parameter_names)


def _random_search(names_train, names_valid, y_train, y_valid, parameter_space, args):
    """Perform random search over given hyper-parameter space."""

    def sample_parameters(parameter_space):
        """Sample parameters from the parameter space."""
        parameters = OrderedDict()
        for key, vals in parameter_space.items():
            parameters[key] = choice(vals)  # sample a value randomly
        return parameters

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
            model_name = _construct_model_name(parameters)
            model_path = os.path.join(args['--model-dir'], model_name)
            _LOGGER.info('Model name: {}'.format(model_name))

            _LOGGER.info('Initialize CharLSTM object with the new parameters...')
            model = CharLSTM(**parameters)

            _LOGGER.info('Started the train() method...')
            score = model.train(names_train, y_train, names_valid, y_valid, model_path,
                                int(args['--batch-size']), int(args['--patience']))
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


def _simple_train(names_train, names_valid, y_train, y_valid, args):
    """Simply train a model using hyper-parameters specified in the args."""
    parameters = {
        'embedding_size': int(args['--embed-size']),
        'char_rnn_size': int(args['--char-rnn-size']),
        'word_rnn_size': int(args['--word-rnn-size']),
        'learning_rate': float(args['--learning-rate']),
        'embedding_dropout': float(args['--embed-dropout']),
        'char_rnn_dropout': float(args['--char-rnn-dropout']),
        'word_rnn_dropout': float(args['--word-rnn-dropout']),
    }
    model_name = _construct_model_name(parameters)
    model_path = os.path.join(args['--model-dir'], model_name)

    _LOGGER.info('Initialize CharLSTM object with the new parameters...')
    model = CharLSTM(**parameters)

    _LOGGER.info('Started the train() method...')
    model.train(names_train, y_train, names_valid, y_valid, model_path, int(args['--batch-size']),
                int(args['--patience']), profile=args['--profile'])


def main():
    """CLI for performing model training."""
    args = docopt(__doc__, version=__version__)
    args = _expand_user_path(args)

    if args['--verbose']:
        set_default_log_level(logging.DEBUG)
        set_log_level(_LOGGER, logging.DEBUG)

    if args['--log-path']:
        log_path = args['--log-path']
        set_default_log_path(log_path)
        set_log_path(_LOGGER, log_path)
        set_classifier_log_path(log_path)

    _LOGGER.debug('Configuration:\n{}'.format(args))

    names, y = _get_training_data(args['--train-data-path'])

    # split into train/valid set
    names_train, names_valid, y_train, y_valid = train_test_split(
        names, y, random_state=_RANDOM_STATE, test_size=float(args['--valid-size']))

    if args['random-search']:
        _random_search(
            names_train, names_valid, y_train, y_valid, _get_parameter_space(), args)
    else:
        _simple_train(names_train, names_valid, y_train, y_valid, args)


if __name__ == '__main__':
    main()
