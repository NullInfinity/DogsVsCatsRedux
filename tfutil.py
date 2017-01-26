"""Some useful helper functions for working with NNs in TensorFlow."""
import math
import os

import numpy as np
import tensorflow as tf

from dataset import FLAGS

"""Create a weight variable initalized with a random normal distribution.

The variable name is `weights` (it is assumed the variable is suitably scoped
by a `tf.name_scope`.

Arguments:
    shape   the shape of the variable
    dtype   the datatype of the variable (default: `tf.float32`)
    mean    the mean of the distribution (default: `0`)
    stddev  the standard deviation of the distribution (default: `0.1`)

Returns:
    A `tf.Variable` with the above properties.
"""
def normal_weight_variable(shape, mean=0.0, stddev=0.1, dtype=tf.float32):
    return tf.get_variable(
            name='weights',
            shape=shape,
            dtype=dtype,
            initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev)
            )

"""Create a bias variable initalized with constants.

The variable name is `bias` (it is assumed the variable is suitably scoped
by a `tf.name_scope`.

Arguments:
    shape   the shape of the variable
    dtype   the datatype of the variable (default: `tf.float32`)
    value   the constant value for initialization

Returns:
    A `tf.Variable` with the above properties.
"""
def constant_bias_variable(shape, value=1.0, dtype=tf.float32):
    return tf.get_variable(
            name='bias',
            shape=shape,
            dtype=dtype,
            initializer=tf.constant_initializer(value=value)
            )


"""Add nodes to the graph to compute cross entropy and write it to summary files."""
def loss_op(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    cross_entropy_avg = tf.reduce_mean(cross_entropy, name='xentropy_avg')
    tf.summary.scalar('xentropy_avg', cross_entropy_avg)
    return cross_entropy_avg

"""Add nodes to the graph to compute accuracy and write it to summary files."""
def accuracy_op(logits, labels, name=''):
    name = name + ('_' if name else '') + 'accuracy'
    correct_prediction = tf.equal(tf.greater(logits, 0.5), tf.greater(labels, 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=name)
    tf.summary.scalar(name=name, tensor=accuracy)
    return accuracy

"""Add nodes to the graph to do training and track the overall training step."""
def train_op(loss, learning_rate):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

"""Run training, write summaries and do evaluation, given computed logits and labels.

Arguments:
    logits              the computed logits
    labels              the known training labels
    valid_accuracy_op   operation to compute model accuracy on validation set
                        (if set to `None`, validation accuracy is not calcuated)
    test_accuracy_op    operation to compute model accuracy on test set
                        (if set to `None`, test accuracy is not calcuated)
    name                the name of the model, used for log and checkpoint directories
    learning_rate       the learning rate to use for training
"""
def run_all(logits, labels, valid_accuracy_op, test_accuracy_op, name, learning_rate):
    loss = loss_op(logits, labels)
    train = train_op(loss, learning_rate=learning_rate)

    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    # create session and summary writer
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    log_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS['LOG_DIR'], name), graph=sess.graph)

    # run ops
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():

            if valid_accuracy_op is not None and step % 5000 == 0:
                accuracy = avg_op(sess, valid_accuracy_op)
                print('Validation accuracy: {acc:.1%}'.format(acc=accuracy))

            if step % 1000 == 0:
                summary, xentropy = sess.run([summary_op, loss])
                saver.save(sess, os.path.join(FLAGS['CHECKPOINT_DIR'], name, name), global_step=step)
                log_writer.add_summary(summary, global_step=step)
                print('Cross Entropy: {xentropy:.2n}'.format(xentropy=xentropy))

            if step % 100 == 0:
                summary = sess.run(summary_op)
                log_writer.add_summary(summary, global_step=step)

            sess.run(train)

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for {} steps.'.format(step))
        saver.save(sess, os.path.join(FLAGS['CHECKPOINT_DIR'], name, name), global_step=step)
        if test_accuracy_op is not None:
            test_accuracy = avg_op(sess, test_accuracy_op)
            print('Test accuracy: {acc:.1%}'.format(acc=test_accuracy))
    finally:
        coord.request_stop()
    coord.join(threads)

    print('Making Kaggle predictions.')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            pass # TODO finish
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()

    sess.close()

"""Evaluate operation `op` over a number of batches.

Arguments:
    sess            the TensorFlow session
    op              the desired operation
    num_examples    total number of examples over which to average

Number of batches is `ceil(num_examples / BATCH_SIZE)
"""
def avg_op(sess, op, num_examples=10000):
    num_batches = int(math.ceil(num_examples / FLAGS['BATCH_SIZE']))
    acc = np.zeros(op.get_shape())
    for i in range(num_batches):
        acc += sess.run(op)
    return acc / num_batches
