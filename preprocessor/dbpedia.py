# -*- coding: UTF-8 -*-
"""
Dbpedia related functions are defined here.
"""
import regex

from chicksexer.constant import POSITIVE_CLASS, NEGATIVE_CLASS

_CLASS_MAP = {'male': POSITIVE_CLASS, 'female': NEGATIVE_CLASS}
_TRIPLE_LINE_REGEX = regex.compile(
    r"^<http://dbpedia\.org/resource/(.*)> <(.*)> (?:\"(.*)\"@[a-z]{2}|<(.*)>) \.$")

__author__ = 'kensk8er'


def gen_triples_from_file(file_path):
    """
    Open the triples file, parse each line and yield them.

    :param file_path: path_to_file which you want to generate triples from
    """
    with open(file_path, encoding='utf-8') as file_:
        for line in file_:
            match_obj = _TRIPLE_LINE_REGEX.search(line)

            if not match_obj:
                continue

            name, _, gender = (string.strip() for string in match_obj.groups() if string)
            yield name.replace('_', ' '), _CLASS_MAP[gender]
