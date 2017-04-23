# coding: utf-8

import input_data
import tensorflow as tf
import numpy as np
import sys
from datetime import datetime


NUM_CATEGORY = 8
EMBEDDING_SIZE = 800  # Hyper parameter
NUM_FILTERS = EMBEDDING_SIZE
FILTER_SIZES = [3, 4, 5]
L2_LAMBDA = 0.0001
NUM_MINI_BATCH = 64
NUM_EPOCHS = 10
CHECKPOINTS_EVERY = 1000
EVALUATE_EVERY = 100


def log(content):
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(time + ': ' + content)
    sys.stdout.flush()

news = input_data.read_data_sets(validation_size=1000, one_hot=True)
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

# TensorBoard
predict = tf.equal(tf.argmax(predict_y, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
loss_sum = tf.summary.scalar('train loss', loss)
accr_sum = tf.summary.scalar('train accuracy', accuracy)
t_loss_sum = tf.summary.scalar('general loss', loss)
t_accr_sum = tf.summary.scalar('general accuracy', accuracy)
saver = tf.train.Saver()

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('Log', sess.graph)
    train_x_length = len(news.train.texts)
    batch_count = int(train_x_length / NUM_MINI_BATCH) + 1

    print('')
    log('Start training.')
    log('     epoch: %d' % NUM_EPOCHS)
    log('mini batch: %d' % NUM_MINI_BATCH)
    log('train data: %d' % train_x_length)
    log(' test data: %d' % len(news.validation.texts))
    log('We will loop %d count per an epoch.' % batch_count)

    # epochs
    for epoch in range(NUM_EPOCHS):
        random_indice = np.random.permutation(train_x_length)

        # mini batch
        log('Start %dth epoch.' % (epoch + 1))
        for i in range(batch_count):
            mini_batch_x = []
            mini_batch_y = []
            for j in range(min(train_x_length - i * NUM_MINI_BATCH, NUM_MINI_BATCH)):
                mini_batch_x.append(news.train.texts[random_indice[i * NUM_MINI_BATCH + j]])
                mini_batch_y.append(news.train.labels[random_indice[i * NUM_MINI_BATCH + j]])

            # training
            _, v1, v2, v3, v4 = sess.run([train, loss, accuracy, loss_sum, accr_sum],
                                         feed_dict={input_x: mini_batch_x, input_y: mini_batch_y, keep: 0.5})
            log('%4dth mini batch complete. LOSS: %f, ACCR: %f' % (i + 1, v1, v2))

            # log for TensorBoard
            current_step = tf.train.global_step(sess, global_step)
            writer.add_summary(v3, current_step)
            writer.add_summary(v4, current_step)

            # save variables
            if current_step % CHECKPOINTS_EVERY == 0:
                saver.save(sess, 'checkpoints/model', global_step=current_step)
                log('Checkout was completed.')

            # evaluate
            if current_step % EVALUATE_EVERY == 0:
                random_test_indice = np.random.permutation(100)
                random_test_x = news.validation.texts[random_test_indice]
                random_test_y = news.validation.labels[random_test_indice]

                v1, v2, v3, v4 = sess.run([loss, accuracy, t_loss_sum, t_accr_sum],
                                          feed_dict={input_x: random_test_x, input_y: random_test_y, keep: 1.0})
                log('Testing... LOSS: %f, ACCR: %f' % (v1, v2))
                writer.add_summary(v3, current_step)
                writer.add_summary(v4, current_step)

        saver.save(sess, 'checkpoints/model_last')
