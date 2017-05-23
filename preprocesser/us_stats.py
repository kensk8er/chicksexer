# -*- coding: UTF-8 -*-
"""
Module used for parsing US Gender Stats files.
"""
import csv
import os
from collections import defaultdict
from glob import glob

import regex

from chicksexer.constant import NEGATIVE_CLASS, POSITIVE_CLASS

_CLASS_MAP = {'M': POSITIVE_CLASS, 'F': NEGATIVE_CLASS}

__author__ = 'kensk8er'


def compute_gender_probas(dir_path, start_year):
    year_prefix = 'yob'
    name2gender2count = defaultdict(lambda: defaultdict(int))
    for file_path in glob(os.path.join(dir_path, '*.txt')):
        year = int(regex.search(r'/{}(\d\d\d\d)'.format(year_prefix), file_path).groups()[0])
        if year < start_year:
            continue

        with open(file_path, encoding='utf8') as file_:
            csv_reader = csv.reader(file_)
            for name, gender, count in csv_reader:
                name2gender2count[name][_CLASS_MAP[gender]] += int(count)

    name2proba = dict()
    for name, gender2count in name2gender2count.items():
        name2proba[name] = float(gender2count[POSITIVE_CLASS]) / (gender2count[POSITIVE_CLASS] +
                                                                  gender2count[NEGATIVE_CLASS])
    return name2proba
