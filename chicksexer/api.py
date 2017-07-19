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
from .classifier import CharLSTM
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


def predict_genders(names: list, return_proba: bool = True, return_attention: bool = False,
                    neutral_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]) -> Union[list, tuple]:
    """
    Predict genders of the given name strings.

    :param names: list of names that you want to predict the gender
    :param return_proba: if True, return probability estimate of the names belonging to each gender
    :param return_attention: if True, return attentions (weight for each word)
    :param neutral_cutoff: if the probability is lower than this threshold for both genders, it
                           returns 'neutral'. [default: 0.8] (only relevant when return_proba=False)
    :return: list of str (male or female) or {'male': male_proba, 'female': female_proba} or
        tuple of aforementioned plus attentions
    """
    global _model
    if not _model:
        _load_model()

    high_cutoff = neutral_cutoff
    low_cutoff = 1. - neutral_cutoff

    try:
        return_value = _model.predict(
            names, return_proba, return_attention, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

        if return_attention:
            predictions, attentions = return_value
        else:
            predictions, attentions = return_value, None

    except UnseenCharacterException as exception:
        message = '{}. Remove the invalid characters from yor inputs.'.format(
            exception.args[0].replace('Unseen', 'Invalid'))
        raise InvalidCharacterException(message)

    predictions = _filter(names, predictions, return_proba)
    if attentions:
        return predictions, attentions
    else:
        return predictions


def _load_model():
    """Load the model."""
    global _model
    _LOGGER.info('Loading model (only required for the initial prediction)...')
    warnings.filterwarnings("ignore", message='Converting sparse IndexedSlices to a dense')
    _model = CharLSTM.load(_MODEL_PATH)


def predict_gender(name: str, return_proba: bool = True, return_attention: bool = False,
                   neutral_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]) -> Union[str, dict, tuple]:
    """
    Predict the gender of the given name string.

    :param name: name string that you want to predict the gender
    :param return_proba: if True, return probability estimate of the name belonging to each gender
    :param return_attention: if True, return attention (weight for each word)
    :param neutral_cutoff: if the probability is lower than this threshold for both genders, it
                           returns 'neutral'. [default: 0.8] (only relevant when return_proba=False)
    :return: str (male or female) or dict of {'male': male_proba, 'female': female_proba} or
        tuple of aforementioned plus attentions
    """
    return_value = predict_genders([name], return_proba, return_attention, neutral_cutoff)

    if return_attention:
        predictions, attentions = return_value
    else:
        predictions, attentions = return_value, None

    if attentions:
        return predictions[0], attentions[0]
    else:
        return predictions[0]


def change_model(model_path: str) -> None:
    """Change the model of chicksexer to the model stored under `model_path`."""
    global _MODEL_PATH, _model
    _MODEL_PATH = model_path
    _load_model()
