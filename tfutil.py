"""Some useful helper functions for working with NNs in TensorFlow."""
import tensorflow as tf


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
    return tf.Variable(
        name='weights',
        dtype=dtype,
        initial_value=tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
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
    return tf.Variable(
        name='bias',
        dtype=dtype,
        initial_value=tf.constant(value=value, shape=shape)
    )
