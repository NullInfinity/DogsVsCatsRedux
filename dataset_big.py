"""
Streams images into the TensorFlow pipeline with optional distortions.
"""

from collections import namedtuple
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
        'CHECKPOINT_DIR':   './checkpoint/',
        'IMAGE_PATTERN':    '*.jpg',

        'TRAIN_SIZE' :      0.8,
        'TEST_SIZE':        0.15,
        # for inception, batch size must be 1
        'BATCH_SIZE':       1,
        }

# for now, train/validation/test split is chosen when module is loaded
def data_files_split():
    files = glob(os.path.join(FLAGS['DATA_DIR'], FLAGS['SRC_DIR'], FLAGS['IMAGE_PATTERN']))

    train_files, valid_files = train_test_split(files, train_size=FLAGS['TRAIN_SIZE'])
    # TODO use FLAGS['TEST_SIZE'] to compute test_size appropriately below
    valid_files, test_files = train_test_split(valid_files, test_size=0.50)

    kaggle_files = glob(os.path.join(FLAGS['DATA_DIR'], FLAGS['KAGGLE_DIR'], FLAGS['IMAGE_PATTERN']))

    return {
        'train': train_files,
        'validation': valid_files,
        'test': test_files,
        'kaggle': kaggle_files,
    }

data_files = data_files_split()

"""The length of flattened image vectors."""
def image_len():
    return FLAGS['IMAGE_SIZE'] ** 2 * FLAGS['IMAGE_CHANNELS']

"""The dimensions of unflattened images.

Arguments:
    include_channels    whether to include the channel dimension
"""
def image_dim(include_channels=False):
    if include_channels:
        return (FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_CHANNELS'])
    return (FLAGS['IMAGE_SIZE'], FLAGS['IMAGE_SIZE'])

"""Return a batch of image, label pairs.

Arguments:
    name        the TFRecord file to read, e.g. `train` or `test`
    display     if `True`, the pixel values are not normalised to [-0.5, 0.5],
                so that the images can be easily displayed on screen.
    batch_size  the size of the batch
    num_epochs  the number of epochs of data to read before terminating
    predict     whether data will be used for training/evaluation or prediction
                if predict=True, the label is either `1` for dog or `0` for cat
                if predict=False, the label is the numerical image ID of each unlabelled image
"""
def inputs(name='train', batch_size=FLAGS['BATCH_SIZE'], num_epochs=1, display=False, predict=False):
    file_list = data_files[name]
    filename_queue = tf.train.string_input_producer(file_list, num_epochs=num_epochs)

    image, label = read_image(filename_queue=filename_queue, predict=predict)

    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=1000+3*batch_size,
        min_after_dequeue=1000,
        num_threads=8
    )
    labels = tf.reshape(labels, [batch_size, 1])

    return images, labels

"""Apply optional distortions to an image."""
def distort(image):
    resize_value = tf.random_uniform([], 1.0, 1.2)

    image = tf.expand_dims(image, 0)

    new_size = tf.cast(
        tf.multiply(
            tf.cast(image_dim(), dtype=tf.float32),
            resize_value
            ),
        dtype=tf.int32
    )
    image = tf.image.resize_bilinear(image, new_size)
    image = tf.random_crop(image, (1,) + image_dim(include_channels=True))

    image = tf.squeeze(image)
    return image

"""Decode JPEG data from a file and distort it if required.

Arguments:
    filename_queue  the TensorFlow queue of filenames

Returns:
    The filename and the (distorted) image as a NumPy array
"""
reader = tf.WholeFileReader()
def read_image(filename_queue, predict):
    key, content = reader.read(filename_queue)

    image = tf.image.decode_jpeg(content, channels=3)
    image = distort(image)
    image = tf.cast(image, dtype=tf.uint8)

    # extract label from filename
    pieces = tf.decode_csv(
        key,
        record_defaults=[['']] * 6,
        field_delim='/',
        )

    # if predict=False, we have labelled images with filenames like
    #   dog.1234.jpg
    # if predict=True, we have unlabelled images with filenames like
    #   1234.jpg
    num_fields = 3
    if predict:
        num_fields = 2

    label = tf.decode_csv(
        pieces[-1],
        record_defaults=[['']] * num_fields,
        field_delim='.',
    )[0]

    return image, label

"""Save a list of images and their labels to a TFRecord file.
Some preprocessing is performed. For details see `read_image`.

Arguments:
    files   the list of input JPEG files
    name    the desired record filename (will have a .tfrecords extension added)
    kaggle  if kaggle=False, the labels is `1` for dog and `0` for cat
            if kaggle=True, the label is the image ID of each unlabelled image
"""
def save_records(files, name, num_epochs, kaggle=False):
    file_queue = tf.train.string_input_producer(
        files,
        num_epochs=num_epochs,
        shuffle=(not kaggle),
    )

    filename, image = sess.run(read_image(file_queue))
    if kaggle:
        # for kaggle set, label means the id of the image
        basename = os.path.splitext(os.path.basename(filename))[0]
        label = int(basename)
    else:
        label = int(b'dog' in filename)
        image_raw = image.tostring()
