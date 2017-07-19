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
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.layers.core import dropout
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.client import timeline

from ._batch import BatchGenerator
from .constant import NEGATIVE_CLASS, NEUTRAL_CLASS, POSITIVE_CLASS, CLASS2DEFAULT_CUTOFF
from ._encoder import CharEncoder
from .util import get_logger, set_log_path as _set_log_path

_TRAIN_PROFILE_FILE = 'profile_train.json'
_VALID_PROFILE_FILE = 'profile_valid.json'
_EMBEDDING_METADATA_FILE = 'metadata.tsv'

_LOGGER = get_logger(__name__)

__author__ = 'kensk8er'


def set_log_path(log_path):
    """Set the log path of the logger in classifier module."""
    _set_log_path(_LOGGER, log_path)


class CharLSTM(object):
    """Character-based language modeling using LSTM."""
    _padding_id = 0  # TODO: 0 is used for actual character as well, which is a bit confusing...
    _checkpoint_file_name = 'model.ckpt'
    _instance_file_name = 'instance.pkl'
    _tensorboard_dir = 'tensorboard.log'

    def __init__(self, embedding_size=32, char_rnn_size=128, word_rnn_size=128, learning_rate=0.001,
                 embedding_dropout=0., char_rnn_dropout=0., word_rnn_dropout=0.):
        # hyper-parameters
        self._embedding_size = embedding_size
        self._char_rnn_size = char_rnn_size
        self._word_rnn_size = word_rnn_size
        self._learning_rate = learning_rate
        self._embedding_dropout = embedding_dropout
        self._char_rnn_dropout = char_rnn_dropout
        self._word_rnn_dropout = word_rnn_dropout

        # other instance variables
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
                X_batch, word_lens, char_lens = self._add_padding(X_batch)
                loss, y_pred = session.run(
                    [nodes['loss'], nodes['y_pred']],
                    feed_dict={nodes['X']: X_batch, nodes['y']: y_batch,
                               nodes['word_lens']: word_lens, nodes['char_lens']: char_lens,
                               nodes['is_train']: False},
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

        _LOGGER.info('Prepare inputs and other variables for the model...')
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

        _LOGGER.info('Building the tensorflow graph...')
        self._build_graph()
        nodes = self._nodes
        session = tf.Session(graph=self._graph)
        summary_writer = tf.summary.FileWriter(
            os.path.join(model_path, self._tensorboard_dir), session.graph)
        self._visualize_embedding(model_path, summary_writer)
        session.run(nodes['init'])
        _LOGGER.info('Start fitting a model...')

        # iterate over batches
        for batch_id, (X_batch, y_batch) in enumerate(train_batch_generator):
            epoch = 1 + iteration // train_size

            if batch_id % summary_interval == 0:
                summaries = session.run(nodes['summaries'])
                summary_writer.add_summary(summaries, global_step=iteration)

            X_batch, word_lens, char_lens = self._add_padding(X_batch)

            # Predict labels and update the parameters
            _, loss, y_pred = session.run(
                [nodes['optimizer'], nodes['loss'], nodes['y_pred']],
                feed_dict={nodes['X']: X_batch, nodes['y']: y_batch, nodes['word_lens']: word_lens,
                           nodes['char_lens']: char_lens, nodes['is_train']: True},
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

    def predict(self, names: list, return_proba=True, return_attention=False,
                low_cutoff=CLASS2DEFAULT_CUTOFF[NEGATIVE_CLASS],
                high_cutoff=CLASS2DEFAULT_CUTOFF[POSITIVE_CLASS]):
        """
        Predict the genders of given names.

        :param names: list of names
        :param return_proba: output probability if set as True
        :param return_attention: if True, return attentions (weights for each time step)
        """
        nodes = self._nodes
        X = self._encode_chars(names)
        X, word_lens, char_lens = self._add_padding(X)
        y_pred, attentions = self._session.run(
            [nodes['y_pred'], nodes['attentions']],
            feed_dict={nodes['X']: X, nodes['word_lens']: word_lens, nodes['char_lens']: char_lens,
                       nodes['is_train']: False})

        # np.ndarray isn't returned when len(X) == 1
        if not isinstance(y_pred, np.ndarray):
            y_pred = [y_pred]

        if return_proba:
            return_value = [{POSITIVE_CLASS: float(proba), NEGATIVE_CLASS: float(1 - proba)}
                            for proba in y_pred]
        else:
            return_value = self._categorize_y(y_pred, low_cutoff, high_cutoff)

        if return_attention:
            return return_value, attentions.tolist()
        else:
            return return_value

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
                # inputs
                nodes['X'] = tf.placeholder(tf.int32, [None, None, None], name='X')
                nodes['y'] = tf.placeholder(tf.float32, [None], name='y')
                nodes['word_lens'] = tf.placeholder(tf.int32, [None], name='word_lens')
                nodes['char_lens'] = tf.placeholder(tf.int32, [None], name='char_lens')
                nodes['is_train'] = tf.placeholder(tf.bool, shape=[], name='is_train')

                # get the shape of the input
                X_shape = tf.shape(nodes['X'])
                batch_size = X_shape[0]
                max_word_len = X_shape[1]
                max_char_len = X_shape[2]

            with tf.name_scope('embedding_layer'):
                nodes['embeddings'] = tf.Variable(
                    tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0),
                    trainable=True, name='embeddings')
                embedded = tf.nn.embedding_lookup(nodes['embeddings'], nodes['X'])
                embedded = dropout(
                    embedded, rate=self._embedding_dropout, training=nodes['is_train'])

            with tf.name_scope('char_rnn_layer') as scope:
                # reshape the embedded matrix in order to pass it to dynamic_rnn
                embedded = tf.reshape(
                    embedded, [batch_size * max_word_len, max_char_len, self._embedding_size])

                char_rnn_fw_cell = LSTMCell(num_units=self._char_rnn_size)
                char_rnn_bw_cell = LSTMCell(num_units=self._char_rnn_size)
                (char_output_fw, char_output_bw), states = tf.nn.bidirectional_dynamic_rnn(
                    char_rnn_fw_cell, char_rnn_bw_cell, embedded, dtype=tf.float32,
                    sequence_length=nodes['char_lens'], scope='{}bidirectional_rnn'.format(scope))

                char_rnn_outputs = tf.concat([char_output_fw, char_output_bw], axis=2)

                with tf.name_scope('char_pooling_layer'):
                    char_rnn_outputs = self._mean_pool(
                        char_rnn_outputs, batch_size, max_char_len, max_word_len,
                        nodes['char_lens'])

                char_rnn_outputs = dropout(
                    char_rnn_outputs, rate=self._char_rnn_dropout, training=nodes['is_train'])

            with tf.name_scope('word_rnn_layer') as scope:
                word_rnn_fw_cell = LSTMCell(num_units=self._word_rnn_size)
                word_rnn_bw_cell = LSTMCell(num_units=self._word_rnn_size)
                (char_output_fw, char_output_bw), states = tf.nn.bidirectional_dynamic_rnn(
                    word_rnn_fw_cell, word_rnn_bw_cell, char_rnn_outputs, dtype=tf.float32,
                    sequence_length=nodes['word_lens'], scope='{}bidirectional_rnn'.format(scope))
                word_rnn_outputs = tf.concat([char_output_fw, char_output_bw], axis=2)

                with tf.name_scope('word_pooling_layer'):
                    word_rnn_outputs, nodes['attentions'] = self._attention_pool(word_rnn_outputs)

                word_rnn_outputs = dropout(
                    word_rnn_outputs, rate=self._word_rnn_dropout, training=nodes['is_train'])

            with tf.variable_scope('softmax_layer'):
                nodes['W_s'] = tf.Variable(
                    tf.random_normal([self._word_rnn_size * 2, 1]), name='weight')
                nodes['b_s'] = tf.Variable(tf.random_normal([1]), name='bias')
                logits = tf.squeeze(tf.matmul(word_rnn_outputs, nodes['W_s']) + nodes['b_s'])
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
        Add padding to X in order to align the sequence lengths.

        :param X: list (each name) of list (each word) of character IDs
        :return: padded list of list of character IDs & list of word length before padding & list of
            character length before padding
        """

        def get_max(X):
            """Compute the maximum word length and maximum character length."""
            max_word_len, max_char_len = 0, 0
            for name in X:
                if max_word_len < len(name):
                    max_word_len = len(name)
                for word in name:
                    if max_char_len < len(word):
                        max_char_len = len(word)
            return max_word_len, max_char_len

        max_word_len, max_char_len = get_max(X)
        word_lens = list()
        char_lens = list()

        for name in X:
            word_lens.append(len(name))
            word_pad_len = max_word_len - len(name)
            name.extend([[] for _ in range(word_pad_len)])

            for word in name:
                char_lens.append(len(word))
                char_pad_len = max_char_len - len(word)
                word.extend([self._padding_id for _ in range(char_pad_len)])

        return X, word_lens, char_lens

    def _encode_chars(self, names, fit=False):
        """Encode list of names into list (each name) of list (each word) of character IDs."""
        if fit:
            name_id2word_id2char_ids = self._encoder.fit_encode(names)
            self._vocab_size = self._encoder.vocab_size
        else:
            name_id2word_id2char_ids = self._encoder.encode(names)
        return name_id2word_id2char_ids

    def _fit_encoder(self, names):
        """Fit the encoder to the given list of names."""
        self._encoder.fit(names)
        self._vocab_size = self._encoder.vocab_size

    def _decode_chars(self, name_id2word_id2char_ids):
        """Decode list (each name) of list (each word) of encoded character IDs into characters."""
        return self._encoder.decode(name_id2word_id2char_ids)

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

    def _mean_pool(self, rnn_outputs, batch_size, max_char_len, max_word_len, char_lens):
        """
        Perform mean-pooling after the character-RNN layer.

        :param rnn_outputs: hidden states of all the time steps after the character-RNN layer
        :return: mean of the hidden states over every time step
        """
        # perform mean pooling over characters
        rnn_outputs = tf.reduce_mean(rnn_outputs, reduction_indices=1)

        # In order to avoid 0 padding affect the mean, multiply by `n / m` where `n` is
        # `max_char_len` and `m` is `char_lens`
        rnn_outputs = tf.multiply(rnn_outputs, tf.cast(max_char_len, tf.float32))  # multiply by `n`

        # swap the dimensions in order to divide by an appropriate value for each time step
        rnn_outputs = tf.transpose(rnn_outputs)

        rnn_outputs = tf.divide(rnn_outputs, tf.cast(char_lens, tf.float32))  # divide by `m`
        rnn_outputs = tf.transpose(rnn_outputs)  # shape back to the original shape

        # batch and word-len dimensions were merged before running character-RNN so shape it back
        rnn_outputs = tf.reshape(rnn_outputs, [batch_size, max_word_len, self._char_rnn_size * 2])

        # there are NaN due to padded words (with char_len=0) so convert those NaN to 0
        rnn_outputs = tf.where(tf.is_nan(rnn_outputs), tf.zeros_like(rnn_outputs), rnn_outputs)

        return rnn_outputs

    def _attention_pool(self, rnn_outputs):
        """
        Perform attention-pooling. Train an attention layer to soft search on hidden states to use
        and return weighted sum of the hidden states.

        :param rnn_outputs: hidden states of all the time steps after the word-RNN layer
        :return: weighted sum of the hidden states and attention weights for each time step
        """
        W = tf.Variable(tf.random_normal([2 * self._word_rnn_size]), name='weight_attention')
        b = tf.Variable(tf.random_normal([1]), name='bias_attention')

        # shape: batch_size * word_len
        attentions = tf.reduce_sum(tf.multiply(W, rnn_outputs), reduction_indices=2) + b
        attentions = tf.nn.softmax(attentions)  # convert to probability

        # swap the dimensions in order to multiply by attentions to each word (the 2nd dimension)
        rnn_outputs = tf.transpose(rnn_outputs, perm=[0, 2, 1])

        # expand the dimension in order to multiply outputs by attentions
        attentions = tf.expand_dims(attentions, axis=1)

        rnn_outputs = tf.multiply(attentions, rnn_outputs)
        rnn_outputs = tf.transpose(rnn_outputs, perm=[0, 2, 1])  # shape back to the original shape

        # pool hidden states of multiple words (after applying attention) into one hidden states
        rnn_outputs = tf.reduce_sum(rnn_outputs, reduction_indices=1)

        return rnn_outputs, tf.squeeze(attentions, axis=1)

    def _visualize_embedding(self, model_path, summary_writer):
        """Create metadata file (and its config file) for tensorboard's embedding visualization."""
        metadata_path = os.path.join(model_path, self._tensorboard_dir, _EMBEDDING_METADATA_FILE)

        # create the metadata config file
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self._nodes['embeddings'].name
        embedding.metadata_path = metadata_path
        projector.visualize_embeddings(summary_writer, config)

        # create metadata file
        with open(metadata_path, 'w', encoding='utf8') as metadata_file:
            metadata_file.write('Character\tID\n')
            for id_, char in enumerate(self._encoder.chars):
                metadata_file.write('{}\t{}\n'.format(char, id_))
