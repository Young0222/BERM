#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attrdict
import copy
import datetime
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.estimator import ModeKeys
import random
import json
import glob

import sys
# distributed
#sys.path.insert(0,'/export/App/training_platform/PinoModel/external/tensorflow/tensorflow/core/custom_ops/python/')
sys.path.insert(0,'/export/sdb/liuziyang7/char_match_gnn/env/tensorflow/tensorflow/core/custom_ops/python/')
#sys.path.insert(0,'/home/admin/chengzhaomeng/tool/stone_Python_3_6_5/bin/python3.6')
import tokenize_fn


class DataConfig(object):
    def __init__(self, config=None):
        #self.train_dataset_files = "/export/sdb/liuziyang7/char_match_gnn/data/char-gnn-train.tsv"
        #self.eval_dataset_files = "/export/sdb/liuziyang7/char_match_gnn/data/char-gnn-eval.tsv"
        #self.test_dataset_files = "/export/sdb/liuziyang7/char_match_gnn/data/char-gnn-eval.tsv"
        self.train_dataset_files = "/export/sdb/liuziyang7/char_match_gnn/data_170m/train_nei.tsv"
        self.eval_dataset_files = "/export/sdb/liuziyang7/char_match_gnn/data_170m/eval_nei.tsv"
        self.test_dataset_files = "/export/sdb/liuziyang7/char_match_gnn/data_170m/eval_nei.tsv"
        self.vocab_word = "vocab.word"
        self.filter_word = "filter.word"
        self.filter_list = [ l.strip() for l in open(self.filter_word, 'r').readlines()]
        self.filter_dict = dict(zip(self.filter_list, [1 for i in range(len(self.filter_list))]))
        self.filter_colsize = 11
        self.batch_size = 1024
        self.num_epochs = 20
        self.perform_shuffle = False
        self.num_word_ids = 40000
        self.query_size = 10
        self.title_size = 65
        self.padding_string = "0_0"

data_config = DataConfig()
feature_names = ["q", "sna", "qiq1_t", "qiq1_q", "qiq2_t", "qiq2_q", "iqi1_q", "iqi1_t", "iqi2_q", "iqi2_t"]


def unigram_and_padding(string_tensor, width, padding_value):
    sparse_tensor = tokenize_fn.unigrams_alphanum_lower_parser(string_tensor)
    return sparse_tensor


def is_in_filter(x):
    if x.decode('utf8') in data_config.filter_dict:
        #print("1:", x.decode('utf8'))
        return np.int32(1)
    else:
        #print("0:", x.decode('utf8'))
        return np.int32(0)

def filter_line(line):
    columns = tf.string_split([line], '\t')
    column_size = tf.size(columns)
    # 1: exist else 0
    query_exists = tf.py_func(is_in_filter, [columns.values[0]], tf.int32)
    #return tf.equal(query_exists, 0)
    return tf.logical_and(tf.equal(query_exists, 0), tf.equal(column_size, data_config.filter_colsize))


def input_fn(filenames, batch_size, num_epochs, perform_shuffle, is_training=False):
    def decode_line(line):
        columns = tf.string_split([line], '\t')
        labels = tf.string_to_number(columns.values[2], out_type=tf.float32)
        query_ids = unigram_and_padding(columns.values[0], data_config.query_size, data_config.padding_string)
        title_ids = unigram_and_padding(columns.values[1], data_config.title_size, data_config.padding_string)
        q_i_q1_title_ids = unigram_and_padding(columns.values[3], data_config.title_size, data_config.padding_string)
        q_i_q1_query_ids = unigram_and_padding(columns.values[4], data_config.query_size, data_config.padding_string)
        q_i_q2_title_ids = unigram_and_padding(columns.values[5], data_config.title_size, data_config.padding_string)
        q_i_q2_query_ids = unigram_and_padding(columns.values[6], data_config.query_size, data_config.padding_string)
        i_q_i1_title_ids = unigram_and_padding(columns.values[7], data_config.title_size, data_config.padding_string)
        i_q_i1_query_ids = unigram_and_padding(columns.values[8], data_config.query_size, data_config.padding_string)
        i_q_i2_title_ids = unigram_and_padding(columns.values[9], data_config.title_size, data_config.padding_string)
        i_q_i2_query_ids = unigram_and_padding(columns.values[10], data_config.query_size, data_config.padding_string)
        return dict(zip(feature_names, [query_ids, title_ids, q_i_q1_title_ids, q_i_q1_query_ids, q_i_q2_title_ids, q_i_q2_query_ids, i_q_i1_title_ids, i_q_i1_query_ids, i_q_i2_title_ids, i_q_i2_query_ids])), labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames)
    if is_training:
        dataset = dataset.filter(filter_line)

    dataset = dataset.map(decode_line, num_parallel_calls=20).prefetch(50000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size*20)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use
    return dataset

def get_all_files(filedir):
    if filedir.startswith("hdfs"):
        if tf.gfile.IsDirectory(filedir):
           return [filedir+'/'+ele for ele in tf.gfile.ListDirectory(filedir)]
        else:
           return [filedir]
    else:
        return glob.glob("%s" % filedir)

def train_input_fn():
    filenames = get_all_files(data_config.train_dataset_files)
    return input_fn(filenames, data_config.batch_size, data_config.num_epochs, data_config.perform_shuffle, True)


def eval_input_fn():
    filenames = get_all_files(data_config.eval_dataset_files)
    return input_fn(filenames, data_config.batch_size, 1, False)

def predict_input_fn():
    filenames = get_all_files(data_config.test_dataset_files)
    return input_fn(filenames, 1, 1, False)


def export_input_fn():
    export_columns = [tf.VarLenFeature(tf.string), tf.VarLenFeature(tf.string)]
    result = dict(zip(feature_names, export_columns))
    return result


def batch_process_mapper(features, config=None):
    for fkey in feature_names:
        untokenizer_tensor = features[fkey]
        if isinstance(untokenizer_tensor, tf.SparseTensor):
           untokenizer_tensor = untokenizer_tensor.values
        if fkey == "q":
            features[fkey] = unigram_and_padding(untokenizer_tensor, data_config.query_size, data_config.padding_string)
        elif fkey == "sna":
            features[fkey] = unigram_and_padding(untokenizer_tensor, data_config.title_size, data_config.padding_string)
    return features


def word2ids(text):
    ## define vocabulary
    vocabulary_feature_column =tf.feature_column.categorical_column_with_vocabulary_file(key="wordstring",
    vocabulary_file=data_config.vocab_word,
    vocabulary_size=None)
    vocab_len = len(open(data_config.vocab_word, 'r').readlines())
    column = tf.feature_column.embedding_column(vocabulary_feature_column, 1, initializer=tf.constant_initializer(np.array([[i] for i in range(vocab_len)])), trainable=False)
    ## map text into ids
    text_str = {"wordstring": tf.reshape(text, [-1])}
    text_ids = tf.cast(tf.feature_column.input_layer(text_str, column), dtype=tf.int32)
    return text_ids


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    x, y = predict_input_fn().make_one_shot_iterator().get_next()
    print("x", x)
    print("y", y)
    z_1, z_2 = word2ids(x["q"].values, x["sna"].values)
    idx = tf.SparseTensor(indices=x["q"].indices,
                          values=tf.reshape(z_1, [-1]), dense_shape=x["q"].dense_shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run([x, y]))
        print(sess.run([z_1, z_2]))
        print(sess.run(idx))
