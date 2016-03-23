
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_ops

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict

if __name__ == "__main__":
    
    batch_size = 10000
    print("batch_size = "+str(batch_size))
    
    # MLP parameters
    num_units = 4096
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # kernel = "baseline"
    kernel = "xnor"
    # kernel = "theano"
    print("kernel = "+ kernel)
    
    print('Loading MNIST dataset...')
    
    test_set = MNIST(which_set= 'test', center = False)
    # Inputs in the range [-1,+1]
    test_set.X = 2* test_set.X.reshape(-1, 784) - 1.
    # flatten targets
    test_set.y = test_set.y.reshape(-1)

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.vector('targets')

    mlp = lasagne.layers.InputLayer(shape=(None, 784),input_var=input)
    
    # Input layer is not binary -> use baseline kernel in first hidden layer
    mlp = binary_ops.DenseLayer(
            mlp,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=num_units,
            kernel = "baseline")               
        
    mlp = lasagne.layers.BatchNormLayer(mlp)
    mlp = lasagne.layers.NonlinearityLayer(mlp,nonlinearity=binary_ops.SignTheano)
    
    for k in range(1,n_hidden_layers):
        
        mlp = binary_ops.DenseLayer(
                mlp,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units,
                kernel = kernel)               
        
        mlp = lasagne.layers.BatchNormLayer(mlp)
        mlp = lasagne.layers.NonlinearityLayer(mlp,nonlinearity=binary_ops.SignTheano)
    
    mlp = binary_ops.DenseLayer(
                mlp, 
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10,
                kernel = kernel)
    
    mlp = lasagne.layers.BatchNormLayer(mlp)
    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), target),dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], test_err)
    
    print("Loading the trained parameters and binarizing the weights...")

    # Load parameters
    with np.load('mnist_parameters.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(mlp, param_values)

    # Binarize the weights
    params = lasagne.layers.get_all_params(mlp)
    for param in params:
        # print param.name
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))
    
    print('Running...')
    
    start_time = time.time()
    
    test_error = val_fn(test_set.X,test_set.y)*100.
    print "test_error = " + str(test_error) + "%"
    
    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")
    