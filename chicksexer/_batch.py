# -*- coding: UTF-8 -*-
"""
Define classes related to batch processing here.
"""
from copy import deepcopy

from sklearn.utils import shuffle

__author__ = 'kensk8er'


class BatchGenerator(object):
    """
    BatchGenerator class cretates a batch iterator on which you can iterate in order to get batches.

    Basic Usage:
        batch_generator = BatchGenerator(X, y, batch_size=128)

        for X_batch, y_batch in batch_generator:
            # it keeps iterating on the batches
            do_something_on_batch(X_batch, y_batch)
    """

    def __init__(self, X, y, batch_size, shuffle=True, valid=False):
        """
        Constructor

        :param X: list of samples (list of lists of char_ids)
        :param y: list of labels (list of probability (of name being a male name))
        :param batch_size: the size of samples in a batch
        :param shuffle: if True, shuffle the data in every new epoch
        :param valid: if True, finish iterating on the data after one pass
        """
        assert isinstance(X, list), 'Invalid argument type type(X) = {}'.format(type(X))
        assert isinstance(y, list), 'Invalid argument type type(y) = {}'.format(type(y))
        assert len(X) == len(y), 'len(X) != len(y)'
        assert batch_size > 0, 'batch_size <= 0'

        # BatchGenerator shouldn't have a by-product
        self._X = deepcopy(X)
        self._y = deepcopy(y)

        self._batch_id = 0
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._valid = valid
        self._data_size = len(self._X)
        self._finish = False

    def __iter__(self):
        return self

    def __next__(self):
        """
        This is called every time you iterate on this object.

        :return: a batch of X, y
        """
        if self._finish:
            raise StopIteration

        X, y = self._gen_batch(self._batch_id, self._batch_size, self._data_size)
        self._batch_id += 1
        return X, y

    def _gen_batch(self, batch_id, batch_size, data_size):
        """Generate batch for given X, y, batch_id, batch_size, and data_size."""
        start_index = (batch_id * batch_size) % data_size
        end_index = ((batch_id + 1) * batch_size) % data_size

        if start_index < end_index:
            return (deepcopy(self._X[start_index: end_index]),
                    deepcopy(self._y[start_index: end_index]))
        else:  # executing here means you have gone over X and y already
            X_first = deepcopy(self._X[start_index:])
            y_first = deepcopy(self._y[start_index:])

            if self._valid:
                self._finish = True
                return X_first, y_first

            # shuffle X and y after going over them if shuffle is True
            if self._shuffle:
                self._X, self._y = shuffle(self._X, self._y)

            X_second = deepcopy(self._X[:end_index])
            y_second = deepcopy(self._y[:end_index])
            return X_first + X_second, y_first + y_second
