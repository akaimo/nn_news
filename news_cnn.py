# coding: utf-8

import input_data
import tensorflow as tf


NUM_CATEGORY = 8
EMBEDDING_SIZE = 128  # Hyper parameter
FILTER_SIZES = [3, 4, 5]

news = input_data.read_data_sets()

# input layer
x_dim = news.train.texts.shape[1]
input_x = tf.placeholder(tf.int32, [None, x_dim])
input_y = tf.placeholder(tf.float32, [None, NUM_CATEGORY])

# Word embedding layer
with tf.name_scope('embedding'):
    w = tf.Variable(tf.random_uniform([len(news.dictionary), EMBEDDING_SIZE], -1.0, 1.0), name='weight')
    e = tf.nn.embedding_lookup(w, input_x)
    ex = tf.expand_dims(e, -1)
