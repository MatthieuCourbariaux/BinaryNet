# Run-time

## Motivations

This subrepository demonstrates the XNOR and baseline GPU kernels described in the article:  
[BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)

## Requirements

* Python 2.7, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html)
* Nvidia GPU (not optional)
* Setting your [Theano flags](http://deeplearning.net/software/theano/library/config.html) to use the GPU
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Downloading MNIST](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/download_mnist.py)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)

##  Matrix multiplication

    nvcc benchmark-cublas.cu -std=c++11 -lcublas && ./a.out
    
This benchmark performs 8192x8192x8192 matrix multiplications with our two kernels and cuBLAS.
The three kernels return exactly the same output when their inputs are constrained to -1 or +1 (but not otherwise).
**The XNOR kernel is about 23x faster than the baseline kernel and 3.4x faster than cuBLAS** on a GTX750 Nvidia GPU.

## MNIST MLP

First, you need to get a trained MNIST MLP:

    python ../Train-time/mnist.py    
    
Then, you can run the trained MNIST MLP using our XNOR GPU kernel:

    python mnist.py
    
The execution time largely depends on your GPU (between 0.4s and 1.5s).
The test error rate should be around 0.96%.
You can compare these results with the baseline kernel or Theano's by modifying the line ~60 in the script.
