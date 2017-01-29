"""Some useful helper functions for working with NNs in TensorFlow."""
import glob
import math
import os

import numpy as np
import tensorflow as tf

from dataset import FLAGS

####### VARIABLE CREATION
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
def weight_variable(shape, mean=0.0, stddev=0.1):
    return tf.get_variable(
            name='weights',
            shape=shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev)
            )

"""Create a weight variable for a convolution initialized with ReLU Xavier initialization.

Arguments:
    shape       the shape of the variable

Initial values are drawn from a normal distribution with mean 0 and variance
  2./shape.
This ensures that the overall layer variance is independent of layer size.
Essentially, this is Xavier initialization, but with an added factor of two,
which has been shown to work well with ReLU nonlinearities. [1]

[1] TODO citation
"""
def xavier_weight_variable(shape):
    size = 1
    for s in shape:
        size *= s
    stddev = math.sqrt(2. / size)
    return weight_variable(shape=shape, mean=0.0, stddev=stddev)

"""Create a weight variable for a convolution initialized with ReLU Xavier initialization.

Arguments:
    size        the width/height of the kernel (kernel is assumed to be square)
    channels    list of channel sizes in the form `[in, out]`

The variable shape will be `[ksize, ksize, channels_in, channels_out]`.
"""
def conv_weight_variable(size, channels):
    shape = [size, size] + channels
    return xavier_weight_variable(shape=shape)

"""Create a weight variable for a fully connected layer initialized with ReLU Xavier initialization.

Arguments:
    size_in     size of previous layer
    size_out    size of current layer

The variable shape will be `[size_in, size_out]`.
"""
def fc_weight_variable(size_in, size_out):
    return xavier_weight_variable(shape=[size_in, size_out])

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
def bias_variable(shape, value=0.1):
    return tf.get_variable(
            name='bias',
            shape=shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(value=value)
            )


######## NETWORK BUILDING
def conv_op(X, size, channels, name, stride, padding='SAME', relu=True):
    with tf.variable_scope(name):
        weights = conv_weight_variable(size=size, channels=channels)
        bias = bias_variable(shape=channels[-1])
        h_conv = tf.nn.conv2d(X, weights, strides=[1, stride, stride, 1], padding=padding) + bias
        if relu:
            h_out = tf.nn.relu(h_conv)
        else:
            h_out = h_conv
    return h_out

class BadPoolMode(Exception):
    pass

def pool_op(X, size, stride, name, padding='SAME', mode='avg'):
    ksize = [1, size, size, 1]
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if mode == 'avg':
            h_pool = tf.nn.avg_pool(X, ksize=ksize, strides=strides, padding=padding, name=name)
        elif mode == 'max':
            h_pool = tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding, name=name)
        else:
            raise BadPoolMode()
    return h_pool

def fc_op(X, channels_in, channels_out, name, relu=True):
    with tf.variable_scope(name):
        weights = fc_weight_variable(channels_in, channels_out)
        bias = bias_variable(channels_out)
        h_fc = tf.matmul(X, weights) + bias
        if relu:
            h_out = tf.nn.relu(h_fc)
        else:
            h_out = h_fc
    return h_out

######## GRAPH BUILDING
"""Add nodes to the graph to compute cross entropy and write it to summary files."""
def loss_op(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    cross_entropy_avg = tf.reduce_mean(cross_entropy, name='xentropy_avg')
    tf.summary.scalar('xentropy_avg', cross_entropy_avg)
    return cross_entropy_avg

"""Add nodes to the graph to compute accuracy and write it to summary files."""
def accuracy_op(logits, labels, name=''):
    name = name + ('_' if name else '') + 'accuracy'
    activation = tf.nn.sigmoid(logits)
    correct_prediction = tf.equal(tf.greater(activation, 0.5), tf.greater(labels, 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=name)
    tf.summary.scalar(name=name, tensor=accuracy)
    return accuracy

"""Add nodes to the graph to do training and track the overall training step."""
def train_op(loss, learning_rate):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

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


######## DIRECTORIES
def get_dir(main, name, pattern):
    return os.path.join(FLAGS[main+'_DIR'], name, '*' if pattern else '')

def log_dir(name, pattern=False):
    return get_dir('LOG', name, pattern=pattern)

def checkpoint_dir(name, pattern=False):
    return get_dir('CHECKPOINT', name, pattern=pattern)

def prediction_file(name):
    return os.path.join(FLAGS['DATA_DIR'], name + '.csv')

def create_if_needed(path):
    # TODO should really check if `path` is an ordinary file and raise an Exception
    if not os.path.isdir(path):
        os.makedirs(path)

######## TRAINING AND EVALUATION
"""Reset graph and remove temporary files, including logs, checkpoints and Kaggle predictions."""
def run_cleanup(name, do_training):
    tf.reset_default_graph()

    if do_training:
        for file in glob.glob(log_dir(name, pattern=True)):
            os.remove(file)
        for file in glob.glob(checkpoint_dir(name, pattern=True)):
            os.remove(file)
    if os.path.isfile(prediction_file(name)):
        os.remove(prediction_file(name))

"""Create log and checkpoint subdirectories."""
def run_setup(name):
    create_if_needed(log_dir(name))
    create_if_needed(checkpoint_dir(name))

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
def run_training(logits, labels, valid_accuracy_op, test_accuracy_op, name, learning_rate):
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

            if valid_accuracy_op is not None and step % 1000 == 0:
                accuracy = avg_op(sess, valid_accuracy_op)
                print('Validation accuracy: {acc:.1%}'.format(acc=accuracy))

            if step % 250 == 0:
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
    sess.close()

"""Make predictions given a logit node in the graph, using the model at its current state of training."""
def run_prediction(logits, name):
    outfile = open(prediction_file(name), 'wb')

    # header
    outfile.write(b'id,label\n')

    activation_op = tf.nn.sigmoid(logits)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir(name))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    sess.run(init_op)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    try:
        while not coord.should_stop():
            dog_prob = sess.run(activation_op).reshape([-1, 1])
            id_column = np.arange(FLAGS['BATCH_SIZE']).reshape([-1, 1]) + step * FLAGS['BATCH_SIZE'] + 1

            np.savetxt(fname=outfile, X=np.append(id_column, dog_prob, 1), fmt=['%i', '%.2f'], delimiter=',')
            outfile.flush()

            step += 1
    except tf.errors.OutOfRangeError:
        print('Wrote {} predictions to {}'.format(step * FLAGS['BATCH_SIZE'], prediction_file(name)))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    outfile.close()

"""Run all operations.

Arguments:
    inference_op    function returning the inference operation (essentially the network graph)
    inputs          function returning input batches
    total_epochs    number of epochs of training data for which to train
    learning_rate   learning rate hyperparameter passed to the optimizer
    name            the name of the network (used as a sub directory for logs, checkpoints, etc)
"""
def run_all(inference_op, inputs, total_epochs, learning_rate, name, do_training=True):
    run_cleanup(name=name, do_training=do_training)
    run_setup(name=name)

    ## TRAINING
    images, labels = inputs(name='train', num_epochs=total_epochs)
    logits = inference_op(images, train=True)

    valid_images, valid_labels = inputs(name='validation', num_epochs=None)
    valid_logits = inference_op(valid_images, train=False)
    valid_accuracy_op = accuracy_op(valid_logits, valid_labels, name='validation')

    test_images, test_labels = inputs(name='test', num_epochs=None)
    test_logits = inference_op(test_images, train=False)
    test_accuracy_op = accuracy_op(test_logits, test_labels, name='test')

    if do_training:
        run_training(logits, labels, valid_accuracy_op, test_accuracy_op, learning_rate=learning_rate, name=name)

    ## PREDICTION
    kaggle_images = inputs(name='kaggle', num_epochs=1, predict=True)
    kaggle_logits = inference_op(kaggle_images, train=False)

    run_prediction(kaggle_logits, name=name)
