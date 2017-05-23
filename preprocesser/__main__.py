# -*- coding: UTF-8 -*-
"""
Main module of preprocesser package. This module will be executed by `python -m preprocesser`.
"""
import os

import pickle

from chicksexer.constant import CLASS2PROB, POSITIVE_CLASS, NEGATIVE_CLASS
from preprocesser import PACKAGE_ROOT
from preprocesser.gender_csv import gen_name_gender_from_csv
from preprocesser.dbpedia import gen_triples_from_file
from preprocesser.us_stats import compute_gender_probas
from preprocesser.util import Name2Proba

_DATA_ROOT = os.path.join(PACKAGE_ROOT, os.path.pardir, 'data')
_RAW_DATA_ROOT = os.path.join(_DATA_ROOT, 'raw')
_PROCESSED_DATA_PATH = os.path.join(_DATA_ROOT, 'name2prob.pkl')

__author__ = 'kensk8er'


def _process_csv(name2probfa):
    """Process csv files that list names and their gender."""
    file_names = ['Black-Female-Names.csv', 'Black-Male-Names.csv', 'Hispanic-Female-Names.csv',
                  'Hispanic-Male-Names.csv', 'Indian-Female-Names.csv', 'Indian-Male-Names.csv',
                  'White-Female-Names.csv', 'White-Male-Names.csv']

    for file_name in file_names:
        for name, gender in gen_name_gender_from_csv(os.path.join(_RAW_DATA_ROOT, file_name)):
            proba = CLASS2PROB[gender]
            name2probfa[name] = proba
    return name2probfa


def _process_dbpedia(name2proba):
    """Process genders_en.ttl downloaded from dbpedia dump."""
    file_name = 'genders_en.ttl'
    for name, gender in gen_triples_from_file(os.path.join(_RAW_DATA_ROOT, file_name)):
        proba = CLASS2PROB[gender]
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
                name2prob[name] = CLASS2PROB[gender]
        return name2prob

    file_name2gender = {
        'male.txt': POSITIVE_CLASS,
        'female.txt': NEGATIVE_CLASS,
    }
    for file_name, gender in file_name2gender.items():
        name2proba = process_common_names(file_name, gender, name2proba)

    return name2proba


def main():
    name2proba = Name2Proba()
    name2proba = _process_dbpedia(name2proba)
    name2proba = _process_csv(name2proba)
    name2proba = _process_us_stats(name2proba)
    name2proba = _process_common_names(name2proba)

    with open(_PROCESSED_DATA_PATH, 'wb') as pickle_file:
        pickle.dump(dict(name2proba), pickle_file)  # save as a normal dict object


if __name__ == '__main__':
    main()
