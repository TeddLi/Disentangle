# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.contrib as contrib

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf
import pdb
import modeling
from modeling import *


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


class DialBertLSTM(object):
    def __init__(self, dial_input_ids, dial_input_masks, dial_token_types, dial_masks, bert_config, is_training,
                 use_one_hot_embeddings):
        self.bert_config = bert_config
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.dial_input_ids = dial_input_ids
        self.dial_input_masks = dial_input_masks
        self.dial_token_types = dial_token_types
        self.dial_masks = dial_masks

        batch_size, max_turns, max_seq_length = get_shape_list(self.dial_input_ids, expected_rank=3)
        dial_input_ids = tf.reshape(self.dial_input_ids, [-1, max_seq_length])
        dial_input_masks = tf.reshape(self.dial_input_masks, [-1, max_seq_length])
        dial_token_types = tf.reshape(self.dial_token_types, [-1, max_seq_length])
        true_turns = tf.reduce_sum(self.dial_masks, 1)


        with tf.variable_scope("sent_encoder"):
            sent_model = modeling.BertModel(config=self.bert_config, is_training=self.is_training,
                                            input_ids=dial_input_ids, input_mask=dial_input_masks,
                                            token_type_ids=dial_token_types,
                                            use_one_hot_embeddings=self.use_one_hot_embeddings)
            sent_arrays = sent_model.get_pooled_output()
            hidden_size = sent_arrays.shape[-1].value
            self.sent_arrays = tf.reshape(sent_arrays, [batch_size, max_turns, hidden_size])

        with tf.variable_scope('dial_encoder'):
            lstm_cell_fw = contrib.rnn.LSTMCell(hidden_size / 2)
            lstm_cell_bw = contrib.rnn.LSTMCell(hidden_size / 2)
            lstm_output, lstm_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw,
                                                                      inputs=self.sent_arrays,
                                                                      sequence_length=true_turns, dtype=tf.float32)
            self.sequence_output = tf.concat(lstm_output, 2)
            # self.sequence_output = layer_norm_and_dropout(self.sequence_output , self.bert_config.hidden_dropout_prob)

    def get_sent_output(self):
        return self.sent_arrays

    def get_dial_output(self):
        return self.sequence_output


class DialModel(object):
    def __init__(self, bert_config, use_one_hot_embeddings, num_turns):
        self.bert_config = bert_config
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.num_turns = num_turns

    def _create_placeholders(self):
        self.dial_input_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='dial_input_ids')
        self.dial_input_masks = tf.placeholder(tf.int32, shape=[None, None, None], name='dial_input_masks')
        self.dial_token_types = tf.placeholder(tf.int32, shape=[None, None, None], name='dial_token_types')
        self.dial_masks = tf.placeholder(tf.int32, shape=[None, None], name='dial_masks')
        self.labels = tf.placeholder(tf.float32, shape=[None, None], name='labels')
        self.target_index = tf.placeholder(tf.int32, shape=[None, None], name='target_index')
        self.candi_indexs = tf.placeholder(tf.int32, shape=[None, None], name='candi_indexs')
        self.data_cluster = tf.placeholder(tf.int32, shape=[None,None],  name = 'data_cluster')
        self.is_training = tf.placeholder(tf.bool, name = "is_training")

    def _build_forward(self):

        encode_model = DialBertLSTM(self.dial_input_ids, self.dial_input_masks, self.dial_token_types,
                                    self.dial_masks, self.bert_config, self.is_training, self.use_one_hot_embeddings)
        # sent_arrays = encode_model.get_sent_output()
        dial_arrays = encode_model.get_dial_output()

        dial_arrays = tf.layers.dropout(dial_arrays, rate=0.1, training=self.is_training)

        self.hidden_size = dial_arrays.shape[-1].value
        target_array = gather_indexes(dial_arrays, self.target_index)
        candi_arrays = gather_indexes(dial_arrays, self.candi_indexs)
        batch_size, num_turns = get_shape_list(self.candi_indexs, expected_rank=2)
        candi_arrays = tf.reshape(candi_arrays, [batch_size, self.num_turns, self.hidden_size])
        target_array = tf.reshape(target_array, [batch_size, 1, self.hidden_size])

        batch_size, num_turns, array_dim = get_shape_list(candi_arrays, expected_rank=3)
        target_arrays = tf.concat([target_array] * self.num_turns, 1)

        candi_arrays = tf.reshape(candi_arrays, [-1, self.hidden_size])
        target_arrays = tf.reshape(target_arrays, [-1, self.hidden_size])
        labels = tf.reshape(self.labels, [-1])
        # pdb.set_trace()
        mix_arrays = tf.concat(
            [target_arrays, candi_arrays, target_arrays * candi_arrays, target_arrays - candi_arrays], axis=-1)
        # mix_arrays = tf.concat([target_arrays, candi_arrays], axis=-1)
        # mix_arrays = tf.nn.tanh(mix_arrays)
        l2_loss = tf.constant(0.0)

        fc_w = tf.get_variable("fc_w", [self.hidden_size * 4, self.hidden_size * 2],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        fc_b = tf.get_variable("fc_b", [self.hidden_size * 2], initializer=tf.zeros_initializer())
        l2_loss += tf.nn.l2_loss(fc_w)
        l2_loss += tf.nn.l2_loss(fc_b)

        mix_arrays = tf.tanh(tf.matmul(mix_arrays, fc_w) + fc_b)
        output_weights = tf.get_variable("output_weights", [1, self.hidden_size * 2],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable("output_bias", [1], initializer=tf.zeros_initializer())
        l2_loss += tf.nn.l2_loss(output_weights)
        l2_loss += tf.nn.l2_loss(output_bias)
        self.l2_loss = l2_loss


        with tf.variable_scope("loss"):
            output_layer = tf.layers.dropout(mix_arrays, rate=0.1, training=self.is_training)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            self.logits = tf.reshape(tf.nn.bias_add(logits, output_bias), [batch_size, self.num_turns])
            self.probabilities = tf.nn.softmax(self.logits, axis=-1)
            nll = -tf.nn.log_softmax(self.logits)
            self.nll = tf.reshape(nll, [1, -1])
            labels = tf.cast(tf.reshape(labels, [1, -1]), 'float32')
            cluster_labels = tf.cast(tf.reshape(self.data_cluster, [1, -1]), 'float32')
            cluster_loss = tf.reduce_mean(tf.matmul(self.nll, cluster_labels, transpose_b=True))/tf.cast(batch_size,'float32')

            self.debug_package = nll, cluster_loss

            self.loss = (tf.reduce_mean(tf.matmul(self.nll, labels, transpose_b=True)) / tf.cast(batch_size,
                                                                                                 'float32')) + 0.00 * self.l2_loss + cluster_loss
            # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits))
            self.preds = tf.cast(tf.argmax(self.probabilities, 1), 'int32')

    def build_graph(self):
        tf.set_random_seed(123)
        self._create_placeholders()
        self._build_forward()
