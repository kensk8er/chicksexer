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

    _start_char = '^'  # the character that represents the start of a name
    _end_char = '$'  # the character that represents the end of a name

    def __init__(self, lower=True):
        self._label_encoder = LabelEncoder()
        self._start_char_id = None
        self._end_char_id = None
        self._fit = False
        self._lower = lower

    def fit(self, samples):
        """
        Fit the character encoder to the samples of characters given.

        :param samples: samples of characters (e.g. sentences)
        """
        characters = ''.join(samples)
        characters = self._clean_characters(characters)
        characters = list(characters)
        characters.insert(0, self._start_char)
        characters.insert(1, self._end_char)
        self._label_encoder.fit(characters)
        self._start_char_id = int(self._label_encoder.transform([self._start_char])[0])
        self._end_char_id = int(self._label_encoder.transform([self._end_char])[0])
        self._fit = True

    def encode(self, samples):
        """
        Encode samples of characters into samples of character IDs using the character encoder.

        :param samples: samples of characters (e.g. sentences)
        :return: Samples of character IDs
        """
        encoded_samples = list()
        for sample in samples:
            sample = self._clean_characters(sample)
            sample = '{}{}{}'.format(self._start_char, sample, self._end_char)

            try:
                encoded_samples.append(self._label_encoder.transform(list(sample)).tolist())
            except ValueError as exception:
                unseen_chars = regex.search(
                    r'y contains new labels: (.*)$', exception.args[0]).groups()[0]
                raise UnseenCharacterException('Unseen characters: {}'.format(unseen_chars))

        return encoded_samples

    def decode(self, samples):
        """
        Decode samples of character IDs into samples of original characters using the character 
        encoder. (Reverse operation of encode())

        :param samples: samples of characters (e.g. sentences)
        :return: Samples of original characters
        """
        decoded_samples = list()
        for sample in samples:
            sample.remove(self._start_char_id)
            sample.remove(self._end_char_id)
            decoded_samples.append(''.join(self._label_encoder.inverse_transform(sample)))
        return decoded_samples

    def fit_encode(self, samples):
        """
        Fit the character encoder to samples of characters and encode them into samples of 
        character IDs using the fitted encoder.

        :param samples: samples of characters (e.g. sentences)
        :return: Samples of character IDs
        """
        self.fit(samples)
        return self.encode(samples)

    @property
    def start_char(self):
        """The character that represents the start of a name."""
        return self._start_char

    @property
    def start_char_id(self):
        """ID of the character that represents the start of a name."""
        return self._start_char_id

    @property
    def end_char(self):
        """The character that represents the end of a name."""
        return self._end_char

    @property
    def end_char_id(self):
        """ID of the character that represents the end of a name."""
        return self._end_char_id

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
