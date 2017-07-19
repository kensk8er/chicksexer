# -*- coding: UTF-8 -*-
"""
Unit tests for chicksexer.api module.
"""
import os
import unittest

from chicksexer import predict_gender, predict_genders
from chicksexer.api import InvalidCharacterException

_TEST_DIR = os.path.dirname(__file__)

__author__ = 'kensk8er'


class ApiTest(unittest.TestCase):
    def test_predict_gender(self):
        # test male names
        with open(os.path.join(_TEST_DIR, 'data', 'male.txt'), 'r', encoding='utf8') as file_:
            for line in file_:
                name = line.strip()

                prediction = predict_gender(name=name, return_proba=True)
                self.assertGreater(
                    prediction['male'], prediction['female'],
                    msg='For name={}, P(male)={} is not greater than P(female)={}'
                        .format(name, prediction['male'], prediction['female']))

                prediction = predict_gender(name=name, return_proba=False)
                self.assertNotEqual(
                    prediction, 'female',
                    msg='For name={}, gender should not be predicted as female.'.format(name))

        # test female names
        with open(os.path.join(_TEST_DIR, 'data', 'female.txt'), 'r', encoding='utf8') as file_:
            for line in file_:
                name = line.strip()

                prediction = predict_gender(name=name, return_proba=True)
                self.assertGreater(
                    prediction['female'], prediction['male'],
                    msg='For name={}, P(female)={} is not greater than P(male)={}'
                        .format(name, prediction['female'], prediction['male']))

                prediction = predict_gender(name=name, return_proba=False)
                self.assertNotEqual(
                    prediction, 'male',
                    msg='For name={}, gender should not be predicted as male.'.format(name))

    def test_predict_genders(self):
        with open(os.path.join(_TEST_DIR, 'data', 'male.txt'), 'r', encoding='utf8') as file_:
            names = [name.strip() for name in file_.readlines()]

            predictions = predict_genders(names, return_proba=True)
            self.assertTrue(
                all(prediction['male'] > prediction['female'] for prediction in predictions))

            predictions = predict_genders(names, return_proba=False)
            self.assertTrue(
                all(prediction != 'female' for prediction in predictions))

        with open(os.path.join(_TEST_DIR, 'data', 'female.txt'), 'r', encoding='utf8') as file_:
            names = [name.strip() for name in file_.readlines()]

            predictions = predict_genders(names, return_proba=True)
            self.assertTrue(
                all(prediction['female'] > prediction['male'] for prediction in predictions))

            predictions = predict_genders(names, return_proba=False)
            self.assertTrue(
                all(prediction != 'male' for prediction in predictions))

    def test_filter(self):
        prediction = predict_gender('@', return_proba=True)
        self.assertEqual(prediction['male'], 0.5)
        self.assertEqual(prediction['female'], 0.5)

        prediction = predict_gender('@', return_proba=False)
        self.assertEqual(prediction, 'neutral')

        prediction = predict_gender('.', return_proba=False)
        self.assertEqual(prediction, 'neutral')

        prediction = predict_gender('', return_proba=False)
        self.assertEqual(prediction, 'neutral')

    def test_unseen_char(self):
        with self.assertRaises(InvalidCharacterException):
            predict_gender('村木', return_proba=True)

    def test_attention(self):
        """Test case for returning attention over words."""
        prediction, attention = predict_gender(
            'Kensuke Muraki', return_proba=False, return_attention=True)
        self.assertEqual(prediction, 'male')
        self.assertGreater(attention[0], attention[1])

        prediction, attention = predict_gender(
            'Kensuke Muraki', return_proba=True, return_attention=True)
        self.assertGreater(prediction['male'], prediction['female'])
        self.assertGreater(attention[0], attention[1])

        predictions, attentions = predict_genders(
            ['Kensuke Muraki', 'Theresa Mary May'], return_proba=False, return_attention=True)
        self.assertEqual(predictions[0], 'male')
        self.assertEqual(predictions[1], 'female')
        self.assertGreater(attentions[0][0], attentions[0][1])
        # the 3rd word is a padded word, which shouldn't have much attention
        self.assertAlmostEqual(attentions[0][2], 0.)
        self.assertGreater(attentions[1][0], attentions[1][2])  # first name must be more important

        predictions, attentions = predict_genders(
            ['Kensuke Muraki', 'Theresa Mary May'], return_proba=True, return_attention=True)
        self.assertGreater(predictions[0]['male'], predictions[0]['female'])
        self.assertGreater(predictions[1]['female'], predictions[1]['male'])
        self.assertGreater(attentions[0][0], attentions[0][1])
        # the 3rd word is a padded word, which shouldn't have much attention
        self.assertAlmostEqual(attentions[0][2], 0.)
        self.assertGreater(attentions[1][0], attentions[1][2])  # first name must be more important


if __name__ == '__main__':
    unittest.main()
