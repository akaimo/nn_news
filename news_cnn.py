# coding: utf-8

import input_data
import tensorflow as tf


NUM_CATEGORY = 8
EMBEDDING_SIZE = 128  # Hyper parameter
NUM_FILTERS = 128
FILTER_SIZES = [3, 4, 5]
L2_LAMBDA = 0.0001

news = input_data.read_data_sets()
keep = tf.placeholder(tf.float32)

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

# Fully-connected layer
with tf.name_scope('fc'):
    total_filters = NUM_FILTERS * len(FILTER_SIZES)
    w = tf.Variable(tf.truncated_normal([total_filters, NUM_CATEGORY], stddev=0.02), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[NUM_CATEGORY]), name='bias')
    h0 = tf.nn.dropout(tf.reshape(p, [-1, total_filters]), keep)
    predict_y = tf.nn.softmax(tf.matmul(h0, w) + b)

# optimize
xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=predict_y))
loss = xentropy + L2_LAMBDA * tf.nn.l2_loss(w)
global_step = tf.Variable(0, name='global_step', trainable=False)
train = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)
