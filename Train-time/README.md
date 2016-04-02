# Train-time

## Motivations

This subrepository enables the reproduction of the benchmark results reported in the article:  
[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)

## Requirements

* Python 2.7, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html)
* A fast Nvidia GPU (or a large amount of patience)
* Setting your [Theano flags](http://deeplearning.net/software/theano/library/config.html) to use the GPU
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Downloading the datasets](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/datasets) you need
* [PyTables](http://www.pytables.org/usersguide/installation.html) (only for the SVHN dataset)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)

## MNIST MLP

    python mnist.py
    
This python script trains an MLP on MNIST with BinaryNet.
It should run for about 6 hours on a Titan Black GPU.
The final test error should be around **0.96%**.

## CIFAR-10 ConvNet

    python cifar10.py
    
This python script trains a ConvNet on CIFAR-10 with BinaryNet.
It should run for about 23 hours on a Titan Black GPU.
The final test error should be around **11.40%**.

## SVHN ConvNet

    python svhn.py
    
This python script trains a ConvNet on SVHN with BinaryNet.
It should run for about 2 days on a Titan Black GPU.
The final test error should be around **2.80%**.
