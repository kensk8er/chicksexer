# -*- coding: UTF-8 -*-
"""
Implement Encoder classes that encode characters into character IDs.
"""
import regex
from sklearn.preprocessing import LabelEncoder

__author__ = 'kensk8er'


class UnseenCharacterException(Exception):
    """Thrown when the encoder tries to encode unseen characters."""


class CharEncoder(object):
    """Encode characters into character IDs."""

    _start_char = '^'  # the character that represents the start of a name word
    _end_char = '$'  # the character that represents the end of a name word
    _separator = ' '  # the character that separates words in a name

    def __init__(self, lower=True):
        self._label_encoder = LabelEncoder()
        self._start_char_id = None
        self._end_char_id = None
        self._fit = False
        self._lower = lower

    def fit(self, names):
        """
        Fit the character encoder to the samples of names given.

        :param names: list of names
        """
        characters = ''.join(names)
        characters = self._clean_characters(characters).replace(' ', '')
        characters = list(characters)
        characters.insert(0, self._start_char)
        characters.insert(1, self._end_char)
        self._label_encoder.fit(characters)
        self._start_char_id = int(self._label_encoder.transform([self._start_char])[0])
        self._end_char_id = int(self._label_encoder.transform([self._end_char])[0])
        self._fit = True

    def encode(self, names):
        """
        Encode list of names into list of list of character IDs using the character encoder.

        :param names: list of names
        :return: list (each name) of list (each word) of character IDs
        """
        name_id2word_id2char_ids = list()
        for name in names:
            name = self._clean_characters(name)
            word_id2char_ids = list()

            for word in name.split(self._separator):
                word = '{}{}{}'.format(self._start_char, word, self._end_char)
                try:
                    word_id2char_ids.append(self._label_encoder.transform(list(word)).tolist())
                except ValueError as exception:
                    unseen_chars = regex.search(
                        r'y contains new labels: (.*)$', exception.args[0]).groups()[0]
                    raise UnseenCharacterException('Unseen characters: {}'.format(unseen_chars))

            name_id2word_id2char_ids.append(word_id2char_ids)

        return name_id2word_id2char_ids

    def decode(self, name_id2word_id2char_ids):
        """
        Decode list of list of character IDs into list of names using the character encoder.
        (Reverse operation of encode())

        :param name_id2word_id2char_ids: list (each name) of list (each word) of character IDs
        :return: list of names
        """
        names = list()
        for word_id2char_ids in name_id2word_id2char_ids:
            words = list()

            for char_ids in word_id2char_ids:
                char_ids.remove(self._start_char_id)
                char_ids.remove(self._end_char_id)
                words.append(''.join(self._label_encoder.inverse_transform(char_ids)))

            names.append(self._separator.join(words))

        return names

    def fit_encode(self, names):
        """
        Fit the character encoder to list of names and encode them into list of list of character
        IDs using the fitted encoder.

        :param names: list of names
        :return: list (each name) of list (each word) of character IDs
        """
        self.fit(names)
        return self.encode(names)

    @property
    def start_char(self):
        """The character that represents the start of a word."""
        return self._start_char

    @property
    def start_char_id(self):
        """ID of the character that represents the start of a word."""
        return self._start_char_id

    @property
    def end_char(self):
        """The character that represents the end of a word."""
        return self._end_char

    @property
    def end_char_id(self):
        """ID of the character that represents the end of a word."""
        return self._end_char_id

    @property
    def separator_char(self):
        """The character that represents the separator of a name."""
        return self._separator

    @property
    def vocab_size(self):
        """The number of unique characters fitted on the encoder."""
        return len(self._label_encoder.classes_)

    @property
    def is_fit(self):
        """True if the encoder is already fit, else False."""
        return self._fit

    def _clean_characters(self, characters):
        """Clean characters (e.g. convert \t to a space)."""
        if self._lower:
            characters = characters.lower()

        characters = regex.sub(r'\t|\s+|\u200d', ' ', characters)
        characters = regex.sub(r'`', "'", characters)
        characters = regex.sub(r'–', "-", characters)
        return characters

    @property
    def chars(self):
        """Characters fitted in order of ascending character ID."""
        return self._label_encoder.classes_
