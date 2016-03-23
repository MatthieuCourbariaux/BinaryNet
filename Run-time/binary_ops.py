
import time

import numpy as np
import theano
import theano.tensor as T

import theano.misc.pycuda_init
from pycuda.compiler import SourceModule
import pycuda.driver as drv

import theano.sandbox.cuda as cuda
from theano.sandbox.cuda.basic_ops import host_from_gpu

import lasagne

# Homemade (and very unoptimized) 
# Theano GPU matrix multiplication operation
# Our 'baseline' kernel
class Gemm(cuda.GpuOp):
    
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__
    
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 2
        assert inp2.ndim == 2

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])
    
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False, False])

    def make_thunk(self, node, storage_map, _, _2):
        
        mod = SourceModule(open("binary_kernels.cu").read())
        gemm_kernel = mod.get_function("gemm")
    
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            
            # inputs
            A = inputs[0][0]
            B = inputs[1][0]
            
            # dimensions
            m = A.shape[0]
            n = A.shape[1]
            k = B.shape[1]
            assert n == B.shape[0] # Otherwise GEMM is impossible
            assert n%16 == 0 # Block size
            
            # output
            output_shape = (m, k)
            C = outputs[0]
            # only allocate if there is no previous allocation of the right size.
            if C[0] is None or C[0].shape != output_shape:
                C[0] = cuda.CudaNdarray.zeros(output_shape)
            
            # Launching GEMM GPU kernel            
            block_size = 16
            block = (block_size,block_size,1)
            grid = (k / block_size+1, m / block_size+1) # better too many blocks than too little
            gemm_kernel(A,B,C[0], np.intc(m), np.intc(n), np.intc(k), block= block, grid=grid)
            
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk
        
gemm = Gemm()

# Our 'XNOR' kernel
class XnorGemm(cuda.GpuOp):
    
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__
    
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 2
        assert inp2.ndim == 2

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])
    
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False, False])

    def make_thunk(self, node, storage_map, _, _2):
        
        mod = SourceModule(open("binary_kernels.cu").read())
        concatenate_rows_kernel = mod.get_function("concatenate_rows_kernel")
        concatenate_cols_kernel = mod.get_function("concatenate_cols_kernel")
        xnor_kernel = mod.get_function("xnor_gemm")
    
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        
        # THIS IS PROBABLY THE PART YOU ARE INTERESTED IN
        def thunk():
            
            # inputs
            A = inputs[0][0]
            B = inputs[1][0]
            
            # dimensions
            m = A.shape[0]
            n = A.shape[1]
            k = B.shape[1]
            assert n == B.shape[0] # Otherwise GEMM is impossible
            assert n%(32*16) == 0 # Concatenation and block size
            
            # output
            output_shape = (m, k)
            C = outputs[0]
            # only allocate if there is no previous allocation of the right size.
            if C[0] is None or C[0].shape != output_shape:
                C[0] = cuda.CudaNdarray.zeros(output_shape)           
            
            # Concatenating the rows of A  
            Ac = drv.mem_alloc(m*n*4/32)
            block_size = 64            
            block = (block_size,1,1)
            grid = (m*n/(block_size*32)+1,1)
            concatenate_rows_kernel(A,Ac, np.intc(m*n/32), block= block, grid=grid)
            
            # Concatenating the columns of B
            Bc = drv.mem_alloc(n*k*4/32)  
            block_size = 64 
            block = (block_size,1,1)
            grid = (k/block_size+1,1)
            concatenate_cols_kernel(B,Bc, np.intc(n), np.intc(k), block= block, grid=grid)
            
            # Launching xnor_kernel
            block_size = 16
            block = (block_size,block_size,1)
            grid = (k / block_size + 1, m / block_size + 1) # better too many blocks than too little
            xnor_kernel(Ac,Bc,C[0], np.intc(m), np.intc(n/32.), np.intc(k), block= block, grid=grid)
            
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk
        
xnor_gemm = XnorGemm()
    
def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)

def SignTheano(x):
    return T.cast(2.*T.ge(x,0)-1., theano.config.floatX)

# A custom Lasagne dense layer using our GPU kernels.
class DenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, kernel="theano", **kwargs):
        
        self.kernel = kernel
        super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
            
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        if self.kernel == "baseline":
            activation = gemm(input, self.W)
        
        if self.kernel == "theano":
            activation = T.dot(input, self.W)
        
        if self.kernel == "xnor":
            activation = xnor_gemm(input, self.W)
        
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
    
# Test suite
if __name__ == "__main__":   
    # N = 8192
    N = 4096
    m = N
    n = N
    k = N
    # m = 784
    # n = 512 
    # k = 10
    
    A = T.fmatrix()
    B = T.fmatrix()
    dot1 = theano.function([A,B], T.dot(A, B))
    dot2 = theano.function([A,B], host_from_gpu(gemm(A, B)))
    dot3 = theano.function([A,B], host_from_gpu(xnor_gemm(A,B)))
    
    # Generating random BINARY matrices
    a = SignNumpy(np.random.randn(m, n))
    b = SignNumpy(np.random.randn(n, k))
    # a = np.float32(np.random.randn(m, n))
    # b = np.float32(np.random.randn(n, k))

    start_time = time.time()
    c1 = dot1(a,b)
    dot1_duration = time.time() - start_time
    # print c1[0][0]
    print("Theano time = "+str(dot1_duration)+"s")
    
    start_time = time.time()
    c3 = dot3(a,b)
    dot3_duration = time.time() - start_time
    # print c3[0][0]
    print("XNOR kernel time = "+str(dot3_duration)+"s")
    
    start_time = time.time()
    c2 = dot2(a,b)
    dot2_duration = time.time() - start_time
    # print c2[0][0]
    print("Baseline kernel time = "+str(dot2_duration)+"s")
    
    # Asserting the kernels are giving the same output
    print "np.mean(np.absolute(c1-c3)) = " + str(np.mean(np.absolute(c1-c3)))
    print "np.mean(np.absolute(c2-c3)) = " + str(np.mean(np.absolute(c2-c3)))
    print "np.allclose(c1, c3) = " + str(np.allclose(c1, c3))
    print "np.allclose(c2, c3) = " + str(np.allclose(c2, c3))

    