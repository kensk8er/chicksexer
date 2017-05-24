# -*- coding: UTF-8 -*-
"""
Define public APIs that are used by users here.
"""
import os
import warnings
from typing import Union

from chicksexer import PACKAGE_ROOT
from chicksexer._classifier import CharLSTM
from chicksexer._util import get_logger

_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'models')
_model = None

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def predict_genders(names: list, return_proba: bool = True) -> list:
    """
    Predict genders of the given name strings.

    :param names: list of names that you want to predict the gender
    :param return_proba: if True, return probability estimate of the names belonging to each gender
    :return: list of str (male or female) or {'male': male_proba, 'female': female_proba} 
    """
    global _model
    if not _model:
        _LOGGER.info('Loading model (only required for the initial prediction)...')
        warnings.filterwarnings("ignore", message='Converting sparse IndexedSlices to a dense')
        _model = CharLSTM.load(_MODEL_PATH)

    return _model.predict(names, return_proba)


def predict_gender(name: str, return_proba: bool = True) -> Union[str, dict]:
    """
    Predict the gender of the given name string.

    :param name: name string that you want to predict the gender
    :param return_proba: if True, return probability estimate of the name belonging to each gender
    :return: str (male or female) or dict of {'male': male_proba, 'female': female_proba} 
    """
    return predict_genders([name], return_proba)[0]
