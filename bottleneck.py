"""Precomputes Inception bottlenecks from TFRecord files.

Running as a script purges tfrecord files before regenerating them.
"""

from glob import glob
import os.path

import tensorflow as tf

import dataset
from inception_v4 import inception_v4, inception_v4_arg_scope
import tfutil as tfu

FLAGS = {
        #'MODEL_FILE':                   './inception/classify_image_graph_def.pb',
        'CHECKPOINT_DIR':               './',
        'CHECKPOINT_FILE':              'inception_v4.ckpt',
        'BOTTLENECK_DIR':               './bottleneck/',
        'BOTTLENECK_TENSOR_NAME':       'pool_3/_reshape:0',
        'BOTTLENECK_SIZE':              1536,
        'RESIZED_INPUT_TENSOR_NAME':    'ResizeBilinear:0',
        'BATCH_SIZE':                   100,
}

def _raw_inputs(name, num_epochs, predict):
    file_queue = tf.train.string_input_producer([os.path.join(FLAGS['BOTTLENECK_DIR'], name+'.tfrecords')], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example, features={
        'bottleneck': tf.FixedLenFeature([FLAGS['BOTTLENECK_SIZE']], tf.float32),
        'label': tf.FixedLenFeature([1], tf.int64),
        })

    label = features['label']
    if not predict:
        label = tf.cast(label, tf.float32)

    bottleneck = features['bottleneck']

    return bottleneck, label

"""Return a batch of bottleneck, label pairs.

Arguments:
    name        the TFRecord file to read, e.g. `train` or `test`
    batch_size  the size of the batch
    num_epochs  the number of epochs of data to read before terminating
    predict     whether data will be used for training/evaluation or prediction
                if predict=True, the label is either `1` for dog or `0` for cat
                if predict=False, the label is the numerical image ID of each unlabelled image
"""
def inputs(name='train', batch_size=FLAGS['BATCH_SIZE'], num_epochs=1, predict=False):
    bottleneck, label = _raw_inputs(name, num_epochs, predict=predict)

    bottlenecks, labels = tf.train.shuffle_batch([bottleneck, label], batch_size=batch_size, capacity=1000+3*batch_size, min_after_dequeue=1000, num_threads=8)

    return bottlenecks, labels

"""Remove any previous bottleneck files."""
def clean_all_bottlenecks():
    record_files = glob(os.path.join(FLAGS['BOTTLENECK_DIR'], '*.tfrecords'))
    for record_file in record_files:
        print('Removing {}'.format(record_file))
        os.remove(record_file)


def _write_bottleneck(sess, step, bottleneck_tensor, label_tensor, writer, **kwargs_unused):
    bottleneck_value, label_value = sess.run([bottleneck_tensor, label_tensor])

    bottleneck_raw = bottleneck_value.flatten().tolist()
    label_raw = label_value.flatten().tolist()

    example = tf.train.Example(features=tf.train.Features(feature={
        'bottleneck': tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_raw)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label_raw)),
    }))

    writer.write(example.SerializeToString())


"""Save a list of bottlenecks and their labels to a TFRecord file.

Arguments:
    name    the desired record filename (will have a .tfrecords extension added)
            input data is also read from a TFRecord file with the same name in
            the data/ folder
"""
def save_bottlenecks(name):
    bottleneck_file = os.path.join(FLAGS['BOTTLENECK_DIR'], name + '.tfrecords')
    print('Saving bottlenecks to {}...'.format(bottleneck_file), end='')
    writer = tf.python_io.TFRecordWriter(bottleneck_file)
    bottleneck_tensor, label_tensor = get_bottlenecks(name)
    kwargs = {
            'bottleneck_tensor': bottleneck_tensor,
            'label_tensor': label_tensor,
            'writer': writer,
            'name': name,
            'checkpoint_path': os.path.join(FLAGS['CHECKPOINT_DIR'], FLAGS['CHECKPOINT_FILE']),
            }
    tfu.run_in_tf(func=_write_bottleneck, after=None, **kwargs)
    print('done.')

"""Save all bottlenecks."""
def save_all_bottlenecks():
    for name in ['train', 'validation', 'test', 'kaggle']:
        save_bottlenecks(name=name)

"""Build an Inception v4 graph using TF-Slim and load a checkpoint."""
def get_bottlenecks(name):
    tf.reset_default_graph()

    inputs, labels = dataset.inputs(name=name, batch_size=1, num_epochs=1, predict=True)
    inputs = tf.reshape(inputs, [1, 299, 299, 3])

    # note about num_classes:
    # we set num_classes=1001 so that the output layer weights are the same size as in the checkpoint
    # otherwise restoring raises an InvalidArgumentError
    # we could of course also use `inception_v4_base` to output the model without the output layer
    # but this way we can still use the final pooling and dropout layers from the original architecture
    # we don't care about the size of num_classes since we are only interested in `PreLogitsFlatten` anyway
    with tf.contrib.slim.arg_scope(inception_v4_arg_scope()):
        _, end_points = inception_v4(
                inputs=inputs,
                num_classes=1001,
                create_aux_logits=False,
                is_training=False,
                dropout_keep_prob=1.0,
                )
    return end_points['PreLogitsFlatten'], labels


# run as script to regenerate TFRecord files
if __name__ == '__main__':
    clean_all_bottlenecks()
    save_all_bottlenecks()
