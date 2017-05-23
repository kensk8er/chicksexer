# -*- coding: UTF-8 -*-
"""
Module used for parsing CSV files of name/gender data.
"""
import csv

from chicksexer.constant import NEUTRAL_CLASS, POSITIVE_CLASS, NEGATIVE_CLASS

_NAME_COLUMN = 'name'
_FIRST_NAME_COLUMN = 'first name'
_LAST_NAME_COLUMN = 'last name'
_GENDER_COLUMN = 'gender'
_CLASS_MAP = {'m': POSITIVE_CLASS, 'f': NEGATIVE_CLASS}

__author__ = 'kensk8er'


class GenderCsvException(Exception):
    """Raised when something's wrong with the gender CSV parsing."""


def gen_name_gender_from_csv(file_path):
    """Yield (name, gender) tuples from the given csv file path."""

    def read_header(header):
        if _NAME_COLUMN in header:
            name_id = header.index(_NAME_COLUMN)
            first_name_id = None
            last_name_id = None
        else:
            name_id = None
            first_name_id = header.index(_FIRST_NAME_COLUMN)
            last_name_id = header.index(_LAST_NAME_COLUMN)

        gender_id = header.index(_GENDER_COLUMN)
        return name_id, first_name_id, last_name_id, gender_id

    with open(file_path, encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        name_id, first_name_id, last_name_id, gender_id = None, None, None, None

        for line_id, line in enumerate(csv_reader):
            line = [element.strip() for element in line]
            if line_id == 0:
                name_id, first_name_id, last_name_id, gender_id = read_header(line)
                continue

            gender = _CLASS_MAP[line[gender_id]]
            if name_id is not None:
                yield line[name_id].title(), gender
            elif first_name_id is not None and last_name_id is not None:
                first_name = line[first_name_id].title()
                last_name = line[last_name_id].title()
                yield first_name, gender
                yield '{} {}'.format(first_name, last_name), gender
                yield last_name, NEUTRAL_CLASS  # you can't tell gender of last name
            else:
                raise GenderCsvException('Neither {} or {} and {} do not exist in the csv file.'
                                         .format(_NAME_COLUMN, _FIRST_NAME_COLUMN, _LAST_NAME_COLUMN))
