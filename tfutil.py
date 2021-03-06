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
by a `tf.name_scope`. The variance is chosen using Xavier initialization
modified for ReLU nonlinearities, see e.g. arXiv:1502.01852 [cs.CV].

Arguments:
    shape   the shape of the variable
    factor  the multiplicative factor to tweak Xavier initialization

Returns:
    A `tf.Variable` with the above properties.
"""
def weight_variable(shape, factor=1.43):
    return tf.get_variable(
            name='weights',
            shape=shape,
            initializer=tf.uniform_unit_scaling_initializer(
                factor=factor,
                dtype=tf.float32)
            )

"""Create a weight variable for a convolution initialized with ReLU Xavier initialization.

Arguments:
    size        the width/height of the kernel (kernel is assumed to be square)
    channels    list of channel sizes in the form `[in, out]`

The variable shape will be `[ksize, ksize, channels_in, channels_out]`.
"""
def conv_weight_variable(size, channels):
    shape = [size, size] + channels
    return weight_variable(shape=shape)

"""Create a weight variable for a fully connected layer initialized with ReLU Xavier initialization.

Arguments:
    size_in     size of previous layer
    size_out    size of current layer

The variable shape will be `[size_in, size_out]`.
"""
def fc_weight_variable(size_in, size_out):
    return weight_variable(shape=[size_in, size_out])

"""Create a bias variable initalized with constants.

The variable name is `bias` (it is assumed the variable is suitably scoped
by a `tf.name_scope`.

Arguments:
    shape   the shape of the variable
    value   the constant value for initialization

Returns:
    A `tf.Variable` with the above properties.
"""
def bias_variable(shape, value=0.1):
    return tf.get_variable(
            name='bias',
            shape=shape,
            initializer=tf.constant_initializer(
                value=value,
                dtype=tf.float32)
            )


######## NETWORK BUILDING
"""Create a convolutional layer with stride `[1,stride,stride,1]`."""
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

"""Exception raised if pooling mode below is set to a value other than `avg` or `max`."""
class BadPoolMode(Exception):
    pass

"""Create a pooling layer with size `[1,size,size,1]` and stride `[1,stride,stride,1]`."""
def pool_op(X, size, stride, name, padding='SAME', mode='max'):
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

# alpha - scale factor for L2 regularisation
"""Create a fully connected layer.

Arguments:
    X               the input tensor
    channels_in     size of the input tensor
    channels_out    size of the output tensor
    name            name of the operation
    reg_terms       dictionary to which regularization terms will be added for
                    use by the loss function
    alpha           scale factor for the regularization term
    relu            if set to False, disables the nonlinearity, e.g. for the
                    output layer
"""
def fc_op(X, channels_in, channels_out, name, reg_terms=None, alpha=0.0, relu=True):
    with tf.variable_scope(name):
        weights = fc_weight_variable(channels_in, channels_out)
        bias = bias_variable(channels_out)
        tf.summary.histogram(name=name+'_weights', values=weights)

        h_fc = tf.matmul(X, weights) + bias
        if relu:
            activation = tf.nn.relu(h_fc)
        else:
            activation = h_fc

        loss_name = name + '_loss'
        reg_term = alpha * tf.nn.l2_loss(weights, name=loss_name)
        if reg_terms is not None:
            reg_terms[name] = reg_term

        tf.summary.histogram(name=name+'_weights', values=activation)

    return activation

######## GRAPH BUILDING
"""Add nodes to the graph to compute cross entropy and write it to summary files.

Also adds any regularization terms to the loss operation before returning it."""
def loss_op(logits, labels, name='', reg_terms=None):
    name = name + ('_' if name else '') + 'xentropy'
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name=name)
    cross_entropy_avg = tf.reduce_mean(cross_entropy, name=name+'_avg')
    tf.summary.scalar(name+'_avg', cross_entropy_avg)

    loss = cross_entropy_avg

    if reg_terms is not None:
        for key in reg_terms:
            loss += reg_terms[key]

    return loss

"""Add nodes to the graph to compute accuracy and write it to summary files."""
def accuracy_op(logits, labels, name=''):
    name = name + ('_' if name else '') + 'accuracy'
    activation = tf.nn.sigmoid(logits)
    correct_prediction = tf.equal(tf.greater(activation, 0.5), tf.greater(labels, 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=name)
    tf.summary.scalar(name=name, tensor=accuracy)
    return accuracy

"""Add nodes to the graph to do training and track the overall training step."""
def train_op(loss, learning_rate, optimizer=tf.train.AdamOptimizer):
    global_step = tf.get_variable(
            name='global_step',
            shape=[],
            initializer=tf.constant_initializer(value=0),
            trainable=False,
            dtype=tf.int32,
            )
    return optimizer(learning_rate).minimize(loss, global_step=global_step)

"""Evaluate operation `op` over a number of batches.

Arguments:
    sess            the TensorFlow session
    op              the desired operation
    num_examples    total number of examples over which to average

Number of batches is `ceil(num_examples / BATCH_SIZE)
"""
def avg_op(sess, op, num_examples=2500):
    num_batches = int(math.ceil(num_examples / FLAGS['BATCH_SIZE']))
    acc = np.zeros(op.get_shape())
    for i in range(num_batches):
        acc += sess.run(op)
    return acc / num_batches


######## DIRECTORIES
def get_dir(main, name, pattern):
    return os.path.join(FLAGS[main+'_DIR'], name, '*' if pattern else '')

"""Returns a named subdirectory of the log directory."""
def log_dir(name, pattern=False):
    return get_dir('LOG', name, pattern=pattern)

"""Returns a named subdirectory of the checkpoint directory."""
def checkpoint_dir(name, pattern=False):
    return get_dir('CHECKPOINT', name, pattern=pattern)

"""Returns a named prediction file."""
def prediction_file(name, clip=False):
    filename = name + ('_clipped' if clip else '') + '.csv'
    return os.path.join(FLAGS['DATA_DIR'], filename)

"""Create a directory if it doesn't already exist."""
def create_if_needed(path):
    # TODO should really check if `path` is an ordinary file and raise an Exception
    if not os.path.isdir(path):
        os.makedirs(path)

######## TRAINING AND EVALUATION
"""Reset graph and remove temporary files, including logs, checkpoints and Kaggle predictions."""
def run_cleanup(name):
    tf.reset_default_graph()

    for file in glob.glob(log_dir(name, pattern=True)):
        os.remove(file)
    for file in glob.glob(checkpoint_dir(name, pattern=True)):
        os.remove(file)
    for clip in [False, True]:
        filename = prediction_file(name, clip=clip)
        if os.path.isfile(filename):
            os.remove(filename)

"""Create log and checkpoint subdirectories."""
def run_setup(name):
    create_if_needed(log_dir(name))
    create_if_needed(checkpoint_dir(name))

"""Run some operation inside a session, starting threads as needed.

Arguments:
    func            function to run in the main loop
    after           function to run after the loop
    name            name for the operation
    checkpoint_path path of a trained checkpoint file to restore
    checkpoint      checkpoint object to restore from (only used if checkpoint_path=None)
    step            the starting global step, 0 used if None
    func_args       keyword args to be passed to `func` and `after`
"""
def run_in_tf(func, after, name, checkpoint_path=None, checkpoint=None, step=None, **func_args):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    try:
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)
    except ValueError: # no variables to save
        saver = None

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    summary_op = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS['LOG_DIR'], name), graph=sess.graph)

    sess.run(init_op)

    # checkpoint_path takes precedence over checkpoint
    if saver is not None:
        if checkpoint_path is not None:
            saver.restore(sess, checkpoint_path)
        elif checkpoint is not None:
            saver.restore(sess, checkpoint.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if step is None:
        step = 0

    kwargs = {
            'sess': sess,
            'saver': saver,
            'log_writer': log_writer,
            'summary_op': summary_op,
            'name': name,
            }

    def after_func(step):
        if after is not None:
            after(step=step, **func_args, **kwargs)

    if func is not None:
        try:
            while not coord.should_stop():
                func(step=step, **func_args, **kwargs)
                step += 1
        except tf.errors.OutOfRangeError:
            after_func(step=step)
        finally:
            coord.request_stop()
    else:
        after_func(step=step)
        coord.request_stop()

    coord.join(threads)
    sess.close()

    return step

"""Run training, write summaries and save checkpoints.

Arguments:
    name            a name for the operation (used to label checkpoints and log files)
    optimizer       the optimizer to use for training
    learning_rate   the learning rate to pass to the optimizer
    step            the starting global step (use this to resume training without
                    producing confusing log files
    inference_op    this function should build the inference graph and return the logits
    reg_terms       a dictionary of regularization terms to pass to the loss function
    num_epochs      the number of epochs of data to train over
    inputs          this function should return batches of data and labels, e.g. `dataset.inputs`

Returns:
    The final global step which can be used to later resume training.
"""
def run_training(name, learning_rate, num_epochs, inference_op, inputs, reg_terms, optimizer, step=None):
    tf.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir(name))

    images, labels = inputs(name='train', num_epochs=num_epochs)
    logits = inference_op(images, reg_terms=reg_terms, train=True)
    loss = loss_op(logits, labels, reg_terms=reg_terms)

    train = train_op(loss, learning_rate=learning_rate, optimizer=optimizer)

    train_images, train_labels = inputs(name='train', num_epochs=None)
    train_logits = inference_op(train_images, reg_terms=reg_terms, train=False, share=True)
    train_accuracy_op = accuracy_op(train_logits, train_labels, name='train')
    train_loss_op = loss_op(train_logits, train_labels, name='train', reg_terms=reg_terms)

    valid_images, valid_labels = inputs(name='validation', num_epochs=None)
    valid_logits = inference_op(valid_images, reg_terms=reg_terms, train=False, share=True)
    valid_accuracy_op = accuracy_op(valid_logits, valid_labels, name='validation')
    valid_loss_op = loss_op(valid_logits, valid_labels, name='valid', reg_terms=reg_terms)

    kwargs = {
            'loss':                 loss,
            'train':                train,
            'reg_terms':            reg_terms,
            'train_accuracy_op':    train_accuracy_op,
            'valid_accuracy_op':    valid_accuracy_op,
            'test_accuracy_op':     None,
            'train_loss_op':        train_loss_op,
            'valid_loss_op':        valid_loss_op,
            'test_loss_op':         None,
            'name':                 name,
            'checkpoint':           checkpoint,
            'step':                 step,
            }

    return run_in_tf(func=_training_func,
              after=_training_after,
              **kwargs
              )

def _print_avg_op(sess, op, label, percent=False):
    if op is not None:
        avg_value = avg_op(sess, op)
        number_format_string = '.1%' if percent else '.3f'
        format_string = '{}: {:' + number_format_string + '}'
        print(format_string.format(label, avg_value))

def _training_func(loss, train, log_writer, summary_op, train_accuracy_op, valid_accuracy_op, train_loss_op, valid_loss_op, sess, saver, step, name, **kwargs_unused):
    if step % 1000 == 0:
        _run_eval(sess=sess, train_accuracy_op=train_accuracy_op, valid_accuracy_op=valid_accuracy_op, test_accuracy_op=None, train_loss_op=train_loss_op, valid_loss_op=valid_loss_op, test_loss_op=None)

    if step % 250 == 0:
        summary, xentropy = sess.run([summary_op, loss])
        saver.save(sess, os.path.join(FLAGS['CHECKPOINT_DIR'], name, name), global_step=step)
        log_writer.add_summary(summary, global_step=step)
        print('Cross Entropy: {xentropy:.3f}'.format(xentropy=xentropy))

    if step % 100 == 0:
        summary = sess.run(summary_op)
        log_writer.add_summary(summary, global_step=step)

    sess.run(train)

def _training_after(sess, saver, step, name, **kwargs):
    print('Done training for {} steps.'.format(step))
    if saver is not None:
        saver.save(sess, os.path.join(FLAGS['CHECKPOINT_DIR'], name, name), global_step=step)
    _run_eval(sess=sess, **kwargs)

"""Make predictions given a logit node in the graph, using the model at its current state of training.

Arguments:
    name            a name for the operation (used to label checkpoints and log files)
    inference_op    this function should build the inference graph and return the logits
    reg_terms       a dictionary of regularization terms to pass to the loss function
    clip            if `clip=True`, output probabilities are clipped to `[0.05,0.95]`
    inputs          this function should return batches of data and labels, e.g. `dataset.inputs`
"""
def run_prediction(name, inference_op, inputs, reg_terms, clip=False):
    tf.reset_default_graph()

    outfile = open(prediction_file(name, clip=clip), 'wb')
    outfile.write(b'id,label\n')

    kaggle_images, kaggle_image_ids = inputs(name='kaggle', num_epochs=1, predict=True)
    kaggle_logits = inference_op(kaggle_images, reg_terms=reg_terms, train=False)

    activation_op = tf.nn.sigmoid(kaggle_logits)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir(name))

    kwargs = {
            'outfile':          outfile,
            'activation_op':    activation_op,
            'image_ids':        kaggle_image_ids,
            'checkpoint':       checkpoint,
            'name':             name,
            'clip':             clip,
            }

    run_in_tf(func=_prediction_func, after=_prediction_after, **kwargs)

    outfile.close()

# helper ops for run_prediction
def _prediction_func(outfile, activation_op, image_ids, sess, saver, step, name, clip, **kwargs_unused):
    dog_prob, image_id = sess.run([activation_op, image_ids])
    if clip:
        np.clip(dog_prob, a_min=0.05, a_max=0.95, out=dog_prob)
    dog_prob = dog_prob.reshape([-1, 1])

    np.savetxt(fname=outfile, X=np.append(image_id, dog_prob, 1), fmt=['%i', '%.2f'], delimiter=',')
    outfile.flush()

def _prediction_after(step, name, clip, **kwargs):
    print('Wrote predictions to {}'.format(prediction_file(name, clip=clip)))

"""Make predictions given a logit node in the graph, using the model at its current state of training.

Arguments:
    name            a name for the operation (used to label checkpoints and log files)
    inference_op    this function should build the inference graph and return the logits
    reg_terms       a dictionary of regularization terms to pass to the loss function
    inputs          this function should return batches of data and labels, e.g. `dataset.inputs`
"""
def run_eval(name, inference_op, reg_terms, inputs):
    tf.reset_default_graph()

    train_images, train_labels = inputs(name='train', num_epochs=None)
    train_logits = inference_op(train_images, reg_terms=reg_terms, train=False)
    train_accuracy_op = accuracy_op(train_logits, train_labels, name='train')
    train_loss_op = loss_op(train_logits, train_labels, name='train', reg_terms=reg_terms)

    valid_images, valid_labels = inputs(name='validation', num_epochs=None)
    valid_logits = inference_op(valid_images, reg_terms=reg_terms, train=False, share=True)
    valid_accuracy_op = accuracy_op(valid_logits, valid_labels, name='validation')
    valid_loss_op = loss_op(valid_logits, valid_labels, name='valid', reg_terms=reg_terms)

    test_images, test_labels = inputs(name='test', num_epochs=None)
    test_logits = inference_op(test_images, reg_terms=reg_terms, train=False, share=True)
    test_accuracy_op = accuracy_op(test_logits, test_labels, name='test')
    test_loss_op = loss_op(test_logits, test_labels, name='test', reg_terms=reg_terms)

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir(name))

    kwargs = {
            'train_accuracy_op': train_accuracy_op,
            'valid_accuracy_op': valid_accuracy_op,
            'test_accuracy_op':  test_accuracy_op,
            'train_loss_op':     train_loss_op,
            'valid_loss_op':     valid_loss_op,
            'test_loss_op':      test_loss_op,
            'checkpoint':        checkpoint,
            'name':              name,
            }

    run_in_tf(func=None, after=_run_eval, **kwargs)

def _run_eval(sess, train_accuracy_op, valid_accuracy_op, test_accuracy_op, train_loss_op, valid_loss_op, test_loss_op, **kwargs):
    _print_avg_op(sess=sess, op=train_accuracy_op, label='Train Accuracy', percent=True)
    _print_avg_op(sess=sess, op=valid_accuracy_op, label='Validation Accuracy', percent=True)
    _print_avg_op(sess=sess, op=test_accuracy_op, label='Test Accuracy', percent=True)
    _print_avg_op(sess=sess, op=train_loss_op, label='Train Loss')
    _print_avg_op(sess=sess, op=valid_loss_op, label='Validation Loss')
    _print_avg_op(sess=sess, op=test_loss_op, label='Test Loss')
