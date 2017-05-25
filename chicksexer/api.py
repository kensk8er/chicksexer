# -*- coding: UTF-8 -*-
"""
Define public APIs that are used by users here.
"""
import os
import warnings
from copy import copy
from typing import Union

import regex

from chicksexer import PACKAGE_ROOT
from ._encoder import UnseenCharacterException
from .constant import POSITIVE_CLASS, NEGATIVE_CLASS, NEUTRAL_CLASS, CLASS2DEFAULT_CUTOFF
from ._classifier import CharLSTM
from .util import get_logger

_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'models')
_model = None

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def _filter(names, predictions, return_proba):
    """Filter bad results."""
    neutral_pred = {POSITIVE_CLASS: 0.5, NEGATIVE_CLASS: 0.5} if return_proba else NEUTRAL_CLASS

    for name_id, name in enumerate(names):
        if not regex.search(r'\w', name):
            predictions[name_id] = copy(neutral_pred)

    return predictions


class InvalidCharacterException(Exception):
    """Thrown when there are invalid characters in the inputs."""


def predict_genders(names: list, return_proba: bool = True,
                    neutral_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]) -> list:
    """
    Predict genders of the given name strings.

    :param names: list of names that you want to predict the gender
    :param return_proba: if True, return probability estimate of the names belonging to each gender
    :param neutral_cutoff: if the probability is lower than this threshold for both genders, it 
                           returns 'neutral'. [default: 0.8] (only relevant when return_proba=False)
    :return: list of str (male or female) or {'male': male_proba, 'female': female_proba} 
    """
    global _model
    if not _model:
        _LOGGER.info('Loading model (only required for the initial prediction)...')
        warnings.filterwarnings("ignore", message='Converting sparse IndexedSlices to a dense')
        _model = CharLSTM.load(_MODEL_PATH)

    high_cutoff = neutral_cutoff
    low_cutoff = 1. - neutral_cutoff

    try:
        predictions = _model.predict(
            names, return_proba, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    except UnseenCharacterException as exception:
        message = '{}. Remove the invalid characters from yor inputs.'.format(
            exception.args[0].replace('Unseen', 'Invalid'))
        raise InvalidCharacterException(message)

    return _filter(names, predictions, return_proba)


def predict_gender(name: str, return_proba: bool = True,
                   neutral_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]) -> Union[str, dict]:
    """
    Predict the gender of the given name string.

    :param name: name string that you want to predict the gender
    :param return_proba: if True, return probability estimate of the name belonging to each gender
    :param neutral_cutoff: if the probability is lower than this threshold for both genders, it 
                           returns 'neutral'. [default: 0.8] (only relevant when return_proba=False)
    :return: str (male or female) or dict of {'male': male_proba, 'female': female_proba} 
    """
    return predict_genders([name], return_proba, neutral_cutoff)[0]
