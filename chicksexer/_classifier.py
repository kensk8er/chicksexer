# -*- coding: UTF-8 -*-
"""
This module implements the core ML algorithm of gender classification.
"""
from copy import copy
import os
import pickle
from time import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.contrib.rnn import DropoutWrapper, LSTMCell, MultiRNNCell
from tensorflow.python.client import timeline

from ._batch import BatchGenerator
from .constant import NEGATIVE_CLASS, NEUTRAL_CLASS, POSITIVE_CLASS, CLASS2DEFAULT_CUTOFF
from ._encoder import CharEncoder
from .util import get_logger

_TRAIN_PROFILE_FILE = 'profile_train.json'
_VALID_PROFILE_FILE = 'profile_valid.json'

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


class CharLSTM(object):
    """Character-based language modeling using LSTM."""
    _padding_id = 0  # TODO: 0 is used for actual character as well, which is a bit confusing...
    _checkpoint_file_name = 'model.ckpt'
    _instance_file_name = 'instance.pkl'
    _tensorboard_dir = 'tensorboard.log'

    def __init__(self, embedding_size=128, rnn_size=256, num_rnn_layers=2, learning_rate=0.001,
                 rnn_dropouts=None, final_dropout=1.0):
        # in order to avoid using mutable object as a default argument
        if rnn_dropouts is None:
            # default is 1.0, which means no dropout
            rnn_dropouts = [1.0 for _ in range(num_rnn_layers)]
        assert len(rnn_dropouts) == num_rnn_layers, 'len(rnn_dropouts) != num_rnn_layers'

        self._embedding_size = embedding_size
        self._rnn_size = rnn_size
        self._num_rnn_layers = num_rnn_layers
        self._learning_rate = learning_rate
        self._rnn_dropouts = rnn_dropouts
        self._final_dropout = final_dropout
        self._nodes = None
        self._graph = None
        self._vocab_size = None
        self._encoder = CharEncoder()
        self._num_params = None
        self._session = None

    def train(self, names_train, y_train, names_valid, y_valid, model_path, batch_size=128,
              patience=1024000, stat_interval=1000, valid_interval=1000, summary_interval=1000,
              valid_batch_size=2048, profile=False):
        """Train a gender classifier on the name/gender pairs."""
        start_time = time()

        def add_metric_summaries(mode, iteration, name2metric):
            """Add summary for metric."""
            metric_summary = tf.Summary()
            for name, metric in name2metric.items():
                metric_summary.value.add(tag='{}_{}'.format(mode, name), simple_value=metric)
            summary_writer.add_summary(metric_summary, global_step=iteration)

        def show_train_stats(epoch, iteration, losses, y_cat, y_cat_pred):
            # compute mean statistics
            loss = np.mean(losses)
            accuracy = accuracy_score(y_cat, y_cat_pred)
            score = accuracy - loss

            _LOGGER.info('Epoch={}, Iter={:,}, Mean Training Loss={:.4f}, Accuracy={:.4f}, '
                         'Accuracy - Loss={:.4f}'.format(epoch, iteration, loss, accuracy, score))
            add_metric_summaries('train', iteration, {'cross_entropy': loss, 'accuracy': accuracy,
                                                      'accuracy - loss': score})
            _LOGGER.info('\n{}'.format(classification_report(y_cat, y_cat_pred, digits=3)))
            return list(), list(), list()

        def validate(epoch, iteration, X, y, best_score, patience):
            """Validate the model on validation set."""
            batch_generator = BatchGenerator(X, y, batch_size=valid_batch_size, valid=True)
            losses, y_cat, y_cat_pred = list(), list(), list()
            for X_batch, y_batch in batch_generator:
                X_batch, seq_lens = self._add_padding(X_batch)
                loss, y_pred = session.run(
                    [nodes['loss'], nodes['y_pred']],
                    feed_dict={nodes['X']: X_batch, nodes['y']: y_batch,
                               nodes['seq_lens']: seq_lens, nodes['is_train']: False},
                    options=run_options, run_metadata=run_metadata)
                losses.append(loss)
                y_cat.extend(self._categorize_y(y_batch))
                y_cat_pred.extend(self._categorize_y(y_pred))

            # compute mean statistics
            loss = np.mean(losses)
            accuracy = accuracy_score(y_cat, y_cat_pred)
            score = accuracy - loss

            _LOGGER.info('Epoch={}, Iter={:,}, Validation Loss={:.4f}, Accuracy={:.4f}, '
                         'Accuracy - Loss={:.4f}'.format(epoch, iteration, loss, accuracy, score))
            add_metric_summaries('valid', iteration, {'cross_entropy': loss, 'accuracy': accuracy,
                                                      'accuracy - loss': score})
            _LOGGER.info('\n{}'.format(classification_report(y_cat, y_cat_pred, digits=3)))

            if score > best_score:
                _LOGGER.info('Best score (Accuracy - Loss) so far, save the model.')
                self._save(model_path, session)
                best_score = score

                if iteration * 2 > patience:
                    patience = iteration * 2
                    _LOGGER.info('Increased patience to {:,}'.format(patience))

            if run_metadata:
                with open(_VALID_PROFILE_FILE, 'w') as file_:
                    file_.write(
                        timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

            return best_score, patience

        # prepare inputs and other variables for the model
        self._fit_encoder(names_train + names_valid)
        X_train = self._encode_chars(names_train)
        X_valid = self._encode_chars(names_valid)
        train_size = len(X_train)
        train_batch_generator = BatchGenerator(X_train, y_train, batch_size)
        best_valid_score = np.float64('-inf')
        losses = list()
        y_cat = list()
        y_cat_pred = list()
        iteration = 0

        # profiler
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if profile else None
        run_metadata = tf.RunMetadata() if profile else None

        # Build and launch the graph
        self._build_graph()
        nodes = self._nodes
        session = tf.Session(graph=self._graph)
        summary_writer = tf.summary.FileWriter(
            os.path.join(model_path, self._tensorboard_dir), session.graph)
        session.run(nodes['init'])
        _LOGGER.info('Start fitting a model...')

        # iterate over batches
        for batch_id, (X_batch, y_batch) in enumerate(train_batch_generator):
            epoch = 1 + iteration // train_size

            if batch_id % summary_interval == 0:
                summaries = session.run(nodes['summaries'])
                summary_writer.add_summary(summaries, global_step=iteration)

            X_batch, seq_lens = self._add_padding(X_batch)

            # Predict labels and update the parameters
            _, loss, y_pred = session.run(
                [nodes['optimizer'], nodes['loss'], nodes['y_pred']],
                feed_dict={nodes['X']: X_batch, nodes['y']: y_batch, nodes['seq_lens']: seq_lens,
                           nodes['is_train']: True},
                options=run_options, run_metadata=run_metadata)

            losses.append(loss)
            y_cat.extend(self._categorize_y(y_batch))
            y_cat_pred.extend(self._categorize_y(y_pred))
            iteration += batch_size

            if run_metadata:
                with open(_TRAIN_PROFILE_FILE, 'w') as file_:
                    file_.write(
                        timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

            if batch_id % stat_interval == 0:
                losses, y_cat, y_cat_pred = show_train_stats(
                    epoch, iteration, losses, y_cat, y_cat_pred)

            if batch_id % valid_interval == 0:
                best_valid_score, patience = validate(
                    epoch, iteration, X_valid, y_valid, best_valid_score, patience)

            if iteration > patience:
                _LOGGER.info('Iteration is more than patience, finish training.')
                break

        _LOGGER.info('Finished fitting the model.')
        _LOGGER.info('Best Validation Score (Accuracy - Cross-entropy Loss): {:.4f}'
                     .format(best_valid_score))

        # close the session
        session.close()

        end_time = time()
        _LOGGER.info('Took {:,} seconds to train the model.'.format(int(end_time - start_time)))
        return best_valid_score

    @classmethod
    def load(cls, model_path):
        """
        Load the model from the saved model directory.

        :param model_path: path to the model directory you want to load the model from.
        :return: instance of the model
        """
        _LOGGER.debug('Started loading the model...')
        # load the instance, set _model_path appropriately
        with open(os.path.join(model_path, cls._instance_file_name), 'rb') as model_file:
            instance = pickle.load(model_file)

        # build the graph and restore the session
        instance._build_graph()
        instance._session = tf.Session(graph=instance._graph)
        instance._session.run(instance._nodes['init'])
        instance._nodes['saver'].restore(
            instance._session, os.path.join(model_path, instance._checkpoint_file_name))

        _LOGGER.debug('Finished loading the model.')
        return instance

    def predict(self, names: list, return_proba=True,
                low_cutoff=CLASS2DEFAULT_CUTOFF[NEGATIVE_CLASS],
                high_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]):
        """
        Predict the genders of given names.

        :param names: list of names
        :param return_proba: output probability if set as True
        """
        nodes = self._nodes
        X = self._encode_chars(names)
        X, seq_lens = self._add_padding(X)
        y_pred = self._session.run(
            nodes['y_pred'],
            feed_dict={nodes['X']: X, nodes['seq_lens']: seq_lens, nodes['is_train']: False})

        # np.ndarray isn't returned when len(X) == 1
        if not isinstance(y_pred, np.ndarray):
            y_pred = [y_pred]

        if return_proba:
            return [{POSITIVE_CLASS: float(proba), NEGATIVE_CLASS: float(1 - proba)}
                    for proba in y_pred]
        else:
            return self._categorize_y(y_pred, low_cutoff, high_cutoff)

    def _save(self, model_path, session):
        """Save the tensorflow session and the instance object of this Python class."""
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save the session
        self._nodes['saver'].save(session, os.path.join(model_path, self._checkpoint_file_name))

        # save the instance
        instance = copy(self)
        instance._graph = None  # _graph is not picklable
        instance._nodes = None  # _nodes is not pciklable
        instance._session = None  # _session is not pciklable
        with open(os.path.join(model_path, self._instance_file_name), 'wb') as pickle_file:
            pickle.dump(instance, pickle_file)

    def _build_graph(self):
        """Build computational graph."""

        def get_num_params():
            """Count the number of trainable parameters."""
            num_params = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                var_num_params = 1
                for dimension in shape:
                    var_num_params *= dimension.value
                num_params += var_num_params
            return num_params

        _LOGGER.debug('Building a computational graph...')
        graph = tf.Graph()
        nodes = dict()

        with graph.as_default():
            with tf.name_scope('inputs'):
                nodes['X'] = tf.placeholder(tf.int32, [None, None], name='X')
                nodes['y'] = tf.placeholder(tf.float32, [None], name='y')
                nodes['seq_lens'] = tf.placeholder(tf.int32, [None], name='seq_lens')
                nodes['is_train'] = tf.placeholder(tf.bool, shape=[], name='is_train')
                rnn_dropouts = tf.where(nodes['is_train'], tf.constant(self._rnn_dropouts),
                                        tf.ones([self._num_rnn_layers]))
                final_dropout = tf.where(
                    nodes['is_train'], tf.constant(self._final_dropout), tf.constant(1.0))

                # get the shape of the input
                X_shape = tf.shape(nodes['X'])
                batch_size = X_shape[0]
                max_seq_len = X_shape[1]

            with tf.name_scope('embedding_layer'):
                nodes['embeddings'] = tf.Variable(
                    tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0),
                    trainable=True, name='embeddings')
                embedded = tf.nn.embedding_lookup(nodes['embeddings'], nodes['X'])

            with tf.name_scope('rnn_layer'):
                cells = list()
                for layer_id in range(self._num_rnn_layers):
                    cell = LSTMCell(num_units=self._rnn_size)
                    cell = DropoutWrapper(cell, input_keep_prob=rnn_dropouts[layer_id])
                    cells.append(cell)

                rnn_cell = MultiRNNCell(cells)
                rnn_activations, states = tf.nn.dynamic_rnn(
                    rnn_cell, embedded, nodes['seq_lens'], dtype=tf.float32)

            with tf.name_scope('pooling_layer'):
                final_indices = nodes['seq_lens'] - 1  # final_index = seq_len - 1
                final_indices = tf.range(0, batch_size) * max_seq_len + final_indices
                final_activation = tf.gather(
                    tf.reshape(rnn_activations, [batch_size * max_seq_len, self._rnn_size]),
                    final_indices)
                final_activation = tf.nn.dropout(
                    final_activation, final_dropout, name='final_dropout')

            with tf.variable_scope('softmax_layer'):
                nodes['W_s'] = tf.Variable(tf.random_normal([self._rnn_size, 1]), name='weight')
                nodes['b_s'] = tf.Variable(tf.random_normal([1]), name='bias')
                logits = tf.squeeze(tf.matmul(final_activation, nodes['W_s']) + nodes['b_s'])
                nodes['y_pred'] = tf.nn.sigmoid(logits)

            with tf.variable_scope('optimizer'):
                nodes['loss'] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=nodes['y']))
                nodes['optimizer'] = tf.train.AdamOptimizer(self._learning_rate).minimize(
                    nodes['loss'])

            # initialize the variables
            nodes['init'] = tf.global_variables_initializer()

            # count the number of parameters
            self._num_params = get_num_params()
            _LOGGER.debug('Total number of parameters = {:,}'.format(self._num_params))

            # generate summaries
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                # having ":" in the name is illegal, so replace to "/"
                tf.summary.histogram(variable.name.replace(':', '/'), variable)
            nodes['summaries'] = tf.summary.merge_all()

            # save the model to checkpoint
            nodes['saver'] = tf.train.Saver()

        self._graph = graph
        self._nodes = nodes

    def _add_padding(self, X):
        """
        Add paddings to X in order to align the sequence lengths.

        :param X: list of sequences of character IDs
        :return: padded list of sequences of character IDs & list of sequence length before padding
        """
        max_len = max(len(x) for x in X)
        seq_lens = list()

        for x in X:
            seq_lens.append(len(x))
            pad_len = max_len - len(x)
            x.extend([self._padding_id for _ in range(pad_len)])
        return X, seq_lens

    def _encode_chars(self, samples, fit=False):
        """Convert samples of characters into encoded characters (character IDs)."""
        if fit:
            encoded_samples = self._encoder.fit_encode(samples)
            self._vocab_size = self._encoder.vocab_size
        else:
            encoded_samples = self._encoder.encode(samples)
        return encoded_samples

    def _fit_encoder(self, samples):
        """Fit the encoder to the given samples (of list of character IDs)."""
        self._encoder.fit(samples)
        self._vocab_size = self._encoder.vocab_size

    def _decode_chars(self, samples):
        """Convert samples of encoded character IDs into decoded characters."""
        return self._encoder.decode(samples)

    @staticmethod
    def _categorize_y(y, low_cutoff=CLASS2DEFAULT_CUTOFF[NEGATIVE_CLASS],
                      high_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]):
        """Categorize a list of continuous values y into a list of male/neutral/female labels."""

        def categorize_label(label):
            if label < low_cutoff:
                return NEGATIVE_CLASS
            elif label <= high_cutoff:
                return NEUTRAL_CLASS
            else:
                return POSITIVE_CLASS

        return [categorize_label(label) for label in y]
