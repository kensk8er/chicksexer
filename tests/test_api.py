# -*- coding: UTF-8 -*-
"""
Unit tests for chicksexer.api module.
"""
import os
import unittest

from chicksexer import predict_gender, predict_genders

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


if __name__ == '__main__':
    unittest.main()
