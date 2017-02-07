# Dog vs Cats
My experiments with the Kaggle competition [Dogs vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).
Project is structured as follows:

* `dataset.py`: handles preprocessing and queueing of input data
* `bottleneck.py`: handles preprocessing and queueing of Inception v4 bottlenecks
* `tfutil.py`: a collection of helper functions for building, training and evaluating TensorFlow networks
* `DogsVsCats_Intro.ipynb`: introduction, control and basic fully connected network (~66% on test set)
* `DogsVsCats_Conv.ipynb`: convolutional network
* `DogsVsCats_Inception.ipynb`: transfer learning with [Google Inception v4](https://arxiv.org/abs/1602.07261)
  * `inception_v4.py`, `inception_utils.py`: code to build Inception v4 network with TF-Slim, from TF-Slim [models page](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)

### Setup
To work with this project, the [data](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) must be obtained from Kaggle and placed in `data/raw`.
For transfer learning the Inception v4 [checkpoint](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) must also be downloaded and extracted into the project root.
