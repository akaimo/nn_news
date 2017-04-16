# coding: utf-8

import input_data
import tensorflow as tf


NUM_CATEGORY = 8
EMBEDDING_SIZE = 128  # Hyper parameter
NUM_FILTERS = 128
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

# Temporal 1-D convolutional and max-pooling layer
p_array = []
for filter_size in FILTER_SIZES:
    with tf.name_scope('conv-%d' % filter_size):
        w = tf.Variable(tf.truncated_normal([filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS], stddev=0.02), name='weight')
        b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name='bias')
        c0 = tf.nn.conv2d(ex, w, [1, 1, 1, 1], 'VALID')
        c1 = tf.nn.relu(tf.nn.bias_add(c0, b))
        c2 = tf.nn.max_pool(c1, [1, x_dim - filter_size + 1, 1, 1], [1, 1, 1, 1], 'VALID')
        p_array.append(c2)

p = tf.concat(p_array, 3)
