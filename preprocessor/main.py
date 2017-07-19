# -*- coding: UTF-8 -*-
"""
Main module of preprocessor package. Can be executed by `python -m preprocessor`.
"""
import os
import pickle
from random import shuffle

import numpy as np

from chicksexer.constant import POSITIVE_CLASS, NEGATIVE_CLASS, NEUTRAL_CLASS, CLASS2DEFAULT_CUTOFF
from chicksexer.util import get_logger
from preprocessor import PACKAGE_ROOT
from preprocessor.dbpedia import gen_triples_from_file
from preprocessor.gender_csv import gen_name_gender_from_csv
from preprocessor.us_stats import compute_gender_probas
from preprocessor.util import Name2Proba

__author__ = 'kensk8er'

_DATA_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'data')
_RAW_DATA_ROOT = os.path.join(_DATA_ROOT, 'raw')
_PROCESSED_DATA_PATH = os.path.join(_DATA_ROOT, 'name2proba_{}.pkl')
_NEUTRAL_NAME_AUGMENTATION_NUM = 100000
_FEMALE_NAME_AUGMENTATION_NUM = 85000
_TEST_DATA_SIZE = 10000  # the size of the whole dataset is ~400,000
_LOGGER = get_logger(__name__)

_CLASS2PROB = {
    POSITIVE_CLASS: 1.,
    NEUTRAL_CLASS: 0.5,
    NEGATIVE_CLASS: 0.,
}


def _process_csv(name2probfa):
    """Process csv files that list names and their gender."""
    file_names = ['Black-Female-Names.csv', 'Black-Male-Names.csv', 'Hispanic-Female-Names.csv',
                  'Hispanic-Male-Names.csv', 'Indian-Female-Names.csv', 'Indian-Male-Names.csv',
                  'White-Female-Names.csv', 'White-Male-Names.csv']

    for file_name in file_names:
        for name, gender in gen_name_gender_from_csv(os.path.join(_RAW_DATA_ROOT, file_name)):
            proba = _CLASS2PROB[gender]
            name2probfa[name] = proba
    return name2probfa


def _process_dbpedia(name2proba):
    """Process genders_en.ttl downloaded from dbpedia dump."""
    file_name = 'genders_en.ttl'
    for name, gender in gen_triples_from_file(os.path.join(_RAW_DATA_ROOT, file_name)):
        proba = _CLASS2PROB[gender]
        name2proba[name] = proba
    return name2proba


def _process_us_stats(name2proba, start_year=1940):
    """Process yobxxxx.txt files that list first names and their gender."""
    dir_path = os.path.join(_RAW_DATA_ROOT, 'US-Baby-Name-Stats')
    name2proba_stats = compute_gender_probas(dir_path, start_year)
    for name, proba in name2proba_stats.items():
        name2proba.set_fix_item(name, proba)
    return name2proba


def _process_common_names(name2proba):
    """Process male/female.txt files that list common male/female names."""

    def process_common_names(file_name, gender, name2prob):
        with open(os.path.join(_RAW_DATA_ROOT, file_name), encoding='utf8') as file_:
            for line in file_:
                if line.startswith('#') or line.startswith('\n'):
                    continue
                name = line.strip()
                name2prob[name] = _CLASS2PROB[gender]
        return name2prob

    file_name2gender = {
        'male.txt': POSITIVE_CLASS,
        'female.txt': NEGATIVE_CLASS,
    }
    for file_name, gender in file_name2gender.items():
        name2proba = process_common_names(file_name, gender, name2proba)

    return name2proba


def _augment_full_names(name2proba, gender):
    """Augment neutral names"""
    if gender == 'neutral':
        augmentation_num = _NEUTRAL_NAME_AUGMENTATION_NUM
        low_proba = CLASS2DEFAULT_CUTOFF[NEGATIVE_CLASS]
        high_proba = CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]
    elif gender == 'female':
        augmentation_num = _FEMALE_NAME_AUGMENTATION_NUM
        low_proba = float('-inf')
        high_proba = CLASS2DEFAULT_CUTOFF[NEGATIVE_CLASS]
    else:
        raise ValueError('Invalid argument gender={}'.format(gender))

    neutral_names = [name for name, prob in name2proba.items()
                     if low_proba < prob < high_proba and ' ' not in name]
    multiple = augmentation_num // len(neutral_names)

    with open(os.path.join(_DATA_ROOT, 'surname2proba.pkl'), 'rb') as pickle_file:
        surname2proba = pickle.load(pickle_file)
        surnames, surname_probas = list(), list()
        for surname, proba in surname2proba.items():
            surnames.append(surname)
            surname_probas.append(proba)

    for neutral_name in neutral_names:
        proba = name2proba[neutral_name]
        sampled_surnames = np.random.choice(surnames, multiple, p=surname_probas)
        for surname in sampled_surnames:
            full_name = '{} {}'.format(neutral_name, surname)
            name2proba[full_name] = proba

    return name2proba


def main():
    name2proba = Name2Proba()
    _LOGGER.info('Processing Dbpedia...')
    name2proba = _process_dbpedia(name2proba)
    _LOGGER.info('Processing CSVs...')
    name2proba = _process_csv(name2proba)
    _LOGGER.info('Processing US Stats...')
    name2proba = _process_us_stats(name2proba)
    _LOGGER.info('Processing Common Names...')
    name2proba = _process_common_names(name2proba)
    _LOGGER.info('Augmenting Neutral Names...')
    name2proba = _augment_full_names(name2proba, 'neutral')
    _LOGGER.info('Augmenting Female Names...')
    name2proba = _augment_full_names(name2proba, 'female')
    _LOGGER.info('Saving to the pickle files...')

    # randomly split into train/test set
    name2proba = dict(name2proba)
    assert len(name2proba) > _TEST_DATA_SIZE, 'Whole dataset size is not larger than test set size.'
    ids = list(range(len(name2proba)))
    shuffle(ids)
    test_ids = set(ids[:_TEST_DATA_SIZE])
    name2proba_train = dict()
    name2proba_test = dict()

    for id_, (name, proba) in enumerate(name2proba.items()):
        if id_ in test_ids:
            name2proba_test[name] = proba
        else:
            name2proba_train[name] = proba

    # write to pickle files
    with open(_PROCESSED_DATA_PATH.format('train'), 'wb') as train_file:
        pickle.dump(name2proba_train, train_file)
    with open(_PROCESSED_DATA_PATH.format('test'), 'wb') as test_file:
        pickle.dump(name2proba_test, test_file)
    with open(_PROCESSED_DATA_PATH.format('all'), 'wb') as all_file:
        pickle.dump(name2proba, all_file)
