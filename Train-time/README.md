# Train-time

## Motivations

This subrepository enables the reproduction of the benchmark results reported in the article:  
BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.

## Requirements

* Python, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)
* [PyTables](http://www.pytables.org/usersguide/installation.html) (only for the SVHN dataset)
* A fast Nvidia GPU (or a large amount of patience)

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
