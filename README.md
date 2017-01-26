# Dog vs Cats
My experiments with the Kaggle competition [Dogs vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).
Project is structured as follows:

## Under Version Control
* `DogsVsCats_Intro.ipynb`: introduction, control and basic fully connected network (~66% on test set)
* `DogsVsCats_Conv.ipynb`: convolutional network
* `DogsVsCats_Inception.ipynb`: transfer learning with Google Inception-v4
* `dataset.py`: handles preprocessing and queueing of input data
* `tfutil.py`: a collection of helper functions for building, training and evaluating TensorFlow networks

## Not Under Version Control
* `logs/`: logs and summaries from TensorFlow runs
* `checkpoints/`: model checkpoints from TensorFlow runs
* `data/raw`: the original images (downloaded from Kaggle)
* `data/*.tfrecord`: preprocessed images (can be generated with `python3 dataset.py`)
