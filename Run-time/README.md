# Run-time

## Motivations

This subrepository demonstrates the XNOR and baseline GPU kernels described in the article:  
BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.

## Requirements

* Python, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html) (Bleeding edge version)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)
* Nvidia GPU (not optional)

##  Matrix multiplication

    python binary_gemm.py
    
This script performs 4096x4096x4096 matrix-matrix multiplications with our XNOR and baseline GPU kernels.
The two kernels return exactly the same output when their inputs are constrained to -1 or +1 (but not otherwise).
The XNOR kernel is about 15x faster than the baseline kernel on a GTX750 Nvidia GPU.

## MNIST MLP

First, you need to get a trained MNIST MLP:

    python ../Train-time/mnist.py    
    
Then, you can run the trained MNIST MLP using our XNOR GPU kernel:

    python mnist.py
    
The execution time largely depends on your GPU (between 0.5s and 2s).
The test error rate should be around 0.96%.
You can compare these results with the baseline kernel by modifying the 60-61th lines of the script.