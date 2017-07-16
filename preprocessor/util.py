# -*- coding: UTF-8 -*-
"""
Define utility classes/functions here.
"""

__author__ = 'kensk8er'


class Name2Proba(dict):
    """Data Structure for storing mapping from name to gender probability."""

    def __init__(self):
        super(Name2Proba, self).__init__()
        self._fixed_keys = set()
        self._key2count = dict()

    def __setitem__(self, *args, **kwargs):
        """Override dict.__setitem__()."""
        key = args[0]
        val = args[1]
        assert not kwargs, 'Invalid argument kwargs={}'.format(kwargs)
        assert isinstance(key, str), 'Invalid type for key={}'.format(key)
        assert isinstance(val, float), 'Invalid type for val={}'.format(val)

        if not self.__contains__(key):
            self._key2count[key] = 1
            super(Name2Proba, self).__setitem__(key, val)
        else:
            self._key2count[key] += 1
            if key not in self._fixed_keys:
                # take incremental average
                cur_val = self.__getitem__(key)
                new_val = cur_val + (val - cur_val) / self._key2count[key]
                super(Name2Proba, self).__setitem__(key, new_val)

    def set_fix_item(self, key, val):
        """Set item without taking incremental average and fix the value."""
        super(Name2Proba, self).__setitem__(key, val)
        if key in self._key2count:
            self._key2count[key] += 1
        else:
            self._key2count[key] = 1
        self._fixed_keys.add(key)
