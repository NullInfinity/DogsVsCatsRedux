"""Handles dataset preprocessing into TFRecords files.

Based on the TensorFlow reading input howto, among other things.

Running as a script purges tfrecord files before regenerating them.
"""

from glob import glob
import os.path

from sklearn.model_selection import train_test_split
import tensorflow as tf

FLAGS = {
        'IMAGE_SIZE':       299,
        'IMAGE_CHANNELS':   3,
        'DATA_DIR':         './data/',
        'SRC_DIR':          './raw/train/',
        'KAGGLE_DIR':       './raw/test/',
        'LOG_DIR':          './log/',
        'CHECKPOINT_DIR':   './checkpoints/',
        'IMAGE_PATTERN':    '*.jpg',

        'TRAIN_SIZE' :      0.8,
        'TEST_SIZE':        0.15,
        'BATCH_SIZE':       50,
        }

def image_len():
    return FLAGS['IMAGE_SIZE'] ** 2 * FLAGS['IMAGE_CHANNELS']

def image_dim(include_channels=False):
    if include_channels:
        return (FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_CHANNELS'])
    return (FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_SIZE'])

def _raw_inputs(name, num_epochs, predict):
    file_queue = tf.train.string_input_producer([os.path.join(FLAGS['DATA_DIR'], name+'.tfrecords')], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    if predict:
        features = tf.parse_single_example(serialized_example, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            })
        label = None
    else:
        features = tf.parse_single_example(serialized_example, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            })
        label = tf.cast(features['label'], tf.float32)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(image_len())

    return image, label

"""Return a batch of image, label pairs.

Arguments:
    name    the TFRecord file to read, e.g. `train` or `test`
    display if `True`, the pixel values are not normalised to [-0.5, 0.5],
            so that the images can be easily displayed on screen.
"""
def inputs(name='train', batch_size=FLAGS['BATCH_SIZE'], num_epochs=1, display=False, predict=False):
    image, label = _raw_inputs(name, num_epochs, predict=predict)

    # setting display=True disables centering and normalisation so images can be correctly displayed
    if not display:
        image = tf.cast(image, tf.float32) * (1./255) - 0.5

    if predict:
        return tf.train.batch([image], batch_size=batch_size, capacity=1000+3*batch_size, allow_smaller_final_batch=True, num_threads=4)

    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=1000+3*batch_size, min_after_dequeue=1000, allow_smaller_final_batch=True, num_threads=4)
    labels = tf.reshape(labels, [batch_size, 1])
    return images, labels

reader = tf.WholeFileReader()
# read JPEG data from a file, decode it into an image (i.e. numpy array), then resize it appropriately
def read_image(filename_queue):
    key, content = reader.read(filename_queue)
    image = tf.cast(tf.image.resize_images(tf.image.decode_jpeg(content), image_dim()), dtype=tf.uint8)
    return key, image

"""Save a list of images given by `files`, and their labels, to the record file `name.tfrecords`.

Some preprocessing is performed. For details see `read_image`.

Arguments:
    files   the list of input JPEG files
    name    the desired record filename (will have a .tfrecords extension added)
"""
def save_records(files, name, omit_labels=False):
    file_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=False)

    record_file = os.path.join(FLAGS['DATA_DIR'], name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(record_file)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Writing {}...'.format(record_file), end='')
    try:
        while not coord.should_stop():
            filename, image = sess.run(read_image(file_queue))
            label = int(b'dog' in filename)
            image_raw = image.tostring()

            if omit_labels:
                # no need to save height, width and depth as this is known to be 299x299x3
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    }))
            else:
                # no need to save height, width and depth as this is known to be 299x299x3
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    }))

            writer.write(example.SerializeToString())
    except tf.errors.OutOfRangeError:
        print('done.')
    finally:
        coord.request_stop()

    writer.close()

    coord.join(threads)
    sess.close()

"""Remove any previous .tfrecords files."""
def clean_all_records():
    record_files = glob(os.path.join(FLAGS['DATA_DIR'], '*.tfrecords'))
    for record_file in record_files:
        print('Removing {}'.format(record_file))
        os.remove(record_file)

"""Split data into train, validation and test sets and save each set to a .tfrecords file."""
def save_training_records():
    files = glob(os.path.join(FLAGS['DATA_DIR'], FLAGS['SRC_DIR'], FLAGS['IMAGE_PATTERN']))

    train_files, valid_files = train_test_split(files, train_size=FLAGS['TRAIN_SIZE'])
    valid_files, test_files = train_test_split(valid_files, test_size=0.75)

    save_records(valid_files, 'validation')
    save_records(test_files, 'test')
    save_records(train_files, 'train')

"""Save Kaggle test data to a .tfrecords file."""
def save_kaggle_records():
    files = glob(os.path.join(FLAGS['DATA_DIR'], FLAGS['KAGGLE_DIR'], FLAGS['IMAGE_PATTERN']))
    save_records(files, 'kaggle', omit_labels=True)

"""Save all .tfrecords files."""
def save_all_records():
    save_training_records()
    save_kaggle_records()


if __name__ == '__main__':
    clean_all_records()
    save_all_records()
