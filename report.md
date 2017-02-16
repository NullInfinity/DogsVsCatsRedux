# ML MT2016 Project: Dogs vs Cats

[Dogs vs Cats][DogsVsCats] was a Kaggle competition in 2013 challenging Kagglers
to train models to distinguish between images of cats and dogs.  As of 2016,
there is a [new version][DogsVsCatsRedux] of this competition currently running
as a playground for experimentation given the explosion in popularity and
effectiveness of deep learning (such as convolutional networks for image
classification) over the last few years.

## The Data

The input data consists of 12500 labeled images of dogs (`label=1`)
and 12500 of cats (`label=0`), as well as an additional 12500 unlabelled images
for classification and submission to Kaggle for scoring on the public
leaderboard. The images are of variable size, so all have been rescaled to
`299x299` pixels without regard for aspect ratio (under the assumption that it's
hard to stretch cats into dogs or vice versa).  I chose 299 pixels specifically
so the images could later be easily fed into an Inception network for transfer
learning.

I split the labeled data into train, validation and test sets (split
`0.80:0.05:0.15`) and saved it into TensorFlow TFRecord files.  All
preprocessing was also applied to the unlabeled Kaggle data so the trained model
could be easily applied to it later.  Both the preprocessing into TFRecord
files, and feeding data to the model at runtime is handled by `dataset.py`.

The [introductory notebook](Introduction.ipynb) displays some sample images
from the dataset. It also verifies that each dataset contains about half cats
and half dogs (the validation set in particular is quite small).

## The Models

### Convolutional Network

My primary model is a convolutional network with a number of convolutional
layers with mostly small filters stacked into several groups separated by max
pooling layers. Though the specific architecture is different, this is
generally similar to [VGG16/19][VGG]. The model is trained and evaluated in
`Convolution.ipynb`. It model attains about `88%` accuracy and scores
`0.33956` on the Kaggle leaderboard.

### Transfer Learning

To achieve best performance, I train a linear classifier on top of the
penultimate layer of Google's [Inception v4 network][Inception4]. This analysis
is contained in `DogsVsCats_Inception.ipynb`. The model attains about `99.5%`
accuracy and attains a loss of `0.073` on the Kaggle leaderboard.

## General Notes

### Overfitting

While minimizing the cross entropy on the training batches, the accuracy is
computed periodically on the validation set. If the validation error starts to
increase while the training cross entropy is still falling, it is likely that
overfitting is occuring. I do not plot any learning graphs in the notebooks,
but the cross entropy and validation error are recorded with a TensorFlow
`SummarySaver`, so plots can be viewed with TensorBoard. Log output to the
console from training can also be used to roughly track training against
validation loss/error.

For both models, the validation and test loss matched the training loss well
following training. However, the loss on the Kaggle public leaderboard was
usually a little higher (though still satisfactory). This could suggest the
dreaded [public test set overfitting][KaggleOverfitting]. However, this seems
unlikely due to the low number of submissions. On the other hand, the test set
and especially the validation set were quite small, which might be why I failed
to fully protect against overfitting.

The obvious solution would be larger validation and test sets. Given the limited
data available, one could employ cross validation, but since the networks are relatively slow to train, this would have a high computational cost.

There are a number of additional options for improving performance and
generalisation, such as heavier regularization or stopping training earlier. For
more possibilities, see the section below on [future work](#future-work).

### Initialization

Since I use relu nonlinearities, I use a modified version of Xavier
initialization with initial weights for a node drawn from a uniform distribution
of variance $2/n$ where $n$ is the number of inputs of the node.  This
initialization strategy is justified [here][ModifiedXavier].  In TensorFlow this
is handled internally by the `tf.uniform_unit_scaling_initializer`.

### tfutil

Some utility functionality has been placed in `tfutil.py`, which makes building,
training and evaluating networks easier. I wrote this library myself as I worked
on this project, so it is very limited in scope compared to proper TensorFlow
wrappers such as Keras or TF-Slim, but it is sufficient for the current project.
I chose to write my own wrapper instead of using one off the shelf so I could
learn how to use TensorFlow directly.

## Convolutional Network

Convolutional networks are well suited to vision problems and here we use one
with 10 convolutional layers and one fully connected hidden layer. When
designing the architecture, I followed the principle that one should keep adding
layers as long as the network is not overfitting. However, I also keep the
network fairly shallow by modern standards to achieve a manageable training time
on a single GPU.

I initially experienced relatively poor performance due to overfitting despite
dropout and L2 regularization of the fully connected layers. This seemed to be a
result of overly large convolutional filters (this architecture is described at
the end of this notebook). As such, my final network design instead uses several
groups of small-filter convolutional layers stacked with max pooling layers in
between. Though the individual filters are small, over a number of layers they
should be capable of detecting larger complex features in the input images.
Although the specifics are different, my architecture is roughly inspired by
[VGG16/19][VGG], which prefers the use of many of convolutional layers with
small filter sizes.

Dropout is applied after the inputs and before each fully connected layer to
protect against overdependence on any single node and thus to make the network
generalise better to the test data. L2 regularisation is also applied quite
heavily on the fully connected layers.

The network achieves an accuracy of around 89% and generalises well both to the
local test set and the Kaggle test data (see below for details).

## Possible Improvements

To get better performance from the network while preventing overfitting (and
without increasing the depth or layer size), we could try implementing:
* an advanced preprocessing stage adding noise to and transforming the input
  images (e.g. random scaling and cropping) to provide additional artificial
  input data and help make the network more resilient to noisy data during
  evaluation and prediction;
* [batch normalization][BatchNorm] with `tf.nn.batch_normalization` to enable
  increased learning rates and to help with regularization;
* [spatial pyramid pooling][SPP] to allow arbitrary input size instead of
  rescaling all images to `299x299` pixels.

It might also be sensible to change the train/validation/test split from
`16:1:3` to `8:1:1` since the current split provides only `1250` validation
images. This is sufficiently few that validation accuracy cannot be reliably
computed to 3 significant figures. Better still, given the limited amount of
data, cross validation could be used. This also has its drawbacks, however,
since training the network is a slow process.

## Notes

The network possesses a number of hyperparameters, e.g.:
* size of convolutional filters
* number of convolutional filters per layer
* size of fully connected layers
* regularization scale factors for fully connected layers
* learning rate for AdamOptimizer (and indeed the choice of optimizer itself)

Some tuning was performed by hand on the above until a satisfactory architecture
was chosen. Then several runs were performed to optimize convergence and to
avoid overfitting. This experimentation was based on comparisons between the
training and validation accuracy/loss over time; the test set was only used for
final evaluation before submission to Kaggle. There is always a risk that by
repeated evaluation and submission (to Kaggle) followed by model improvement and
hyperparameter tuning, one can inadvertently fit to the test set or even the
unlabelled Kaggle submission data. By making a clear and careful divide between
the data used for tuning (the validation set) and data used for final evaluation
(test set and unlabelled Kaggle data), however, I hope to have avoided this
problem.

## Results

Note that the values in the table correspond to my best run and may not exactly
match the output of `run_eval` below.

### Training Summary | learning rate | epoch count |
|---------------|-------------| | 1e-4          | 10          | | 1e-5 | 10
|

### Scoring | dataset    | accuracy | loss    |
|------------|----------|---------| | train      | 84.9%    | 0.389   | |
validation | 84.8%    | 0.402   | | test       | 85.6%    | 0.382   | | kaggle |
| 0.40675 | | (clipped)  |          | 0.36827 |


# Dogs vs Cats ## Transfer Learning with Inception

In order to achieve greater performance than the convnet without significant
increase in training time, I now take advantage of the image classification
capabilities of Google's [Inception v4](http://arxiv.org/abs/1602.07261),
pretrained on ImageNet. Since ImageNet is such a large dataset, this network
should be able to detect fairly generic features in our images. Moreover, since
Imagenet contains many breeds of cat and dog as classes, it may even have learnt
features specific to identifying cats or dogs.

I start by computing the bottlenecks (i.e. the penultimate layer activations)
for Inception on our images and then use a single layer (i.e. logistic
regression) on these computed features to predict the probabilty that a given
image contains a dog. Once the bottlenecks are computed and saved to TFRecord
files, they can be fed into the network by an input pipeline just like the
images themselves were for the convnet. Where `dataset.py` handled the image
TFRecords (both reading and writing), `bottleneck.py` handles the bottleneck
TFRecords.

Unlike some older Inception networks, Google does not provide a ProtoBuf
GraphDef file for Inception v4. Instead, Python source files are provided which
recreate the network using
[TF-Slim][TFSlim],
a high level wrapper for TensorFlow. Also provided, of course, are checkpoint
files containing the pretrained weights, biases and other variables. The
following files were downloaded from the TF-Slim [models
page][TFSlimModels]
in order to recreate the Inception v4 network:

* `inception_v4.py`: the main model building file; I modified this slightly for
  compatibility with Python 3 and the latest TensorFlow (see comments in file
  for details)
* `inception_utils.py`: helper functions needed by `inception_v4.py`
* `inception_v4.ckpt`: the model checkpoint

Since the bottleneck creation is handled by the `bottleneck.py` script, all that
remains to do is train a logistic regression model on the cached bottleneck
values. I could use, say, `scikit-learn` for this, providing access to the best
routines for training linear models, as well as alternative linear classifiers
such as SVMs. However, in the name of simplicity and consistency, I instead
treat the linear model as a fully connected network with no hidden layer, so
that it can be trained using TensorFlow and my `tfutil` helper functions as in
the previous notebook.

### Results The results are indeed much better than with the basic convnet.

#### Training Summary
| learning rate | epoch count |
|---------------|-------------|
| 1e-4          | 25          |

#### Scoring
| dataset    | accuracy | loss    |
|------------|----------|---------|
| train      | 99.6%    | 0.031   |
| validation | 99.5%    | 0.033   |
| test       | 99.7%    | 0.033   |
| kaggle     |          | 0.0733  |

## Future Work

Additionally, to help the model generalise, it could be useful to apply random
distortions and noise to the training images on each iteration. This would
require redesigning my input pipeline so I have omitted it for now, but it may
be worth experimenting with in the future.



[DogsVsCats]:         https://www.kaggle.com/c/dogs-vs-cats
[DogsVsCatsRedux]:    https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
[VGG]:                https://arxiv.org/abs/1409.1556
[Inception4]:         http://arxiv.org/abs/1602.07261
[ModifiedXavier]:     https://arxiv.org/abs/1502.01852
[BatchNorm]:          https://arxiv.org/abs/1502.03167
[SPP]:                https://arxiv.org/abs/1406.4729
[TFSlim]:             https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
[TFSlimModels]:       https://github.com/tensorflow/models/tree/master/slim#pre-trained-models
[KaggleOverfitting]:  http://blog.mrtz.org/2015/03/09/competition.html
