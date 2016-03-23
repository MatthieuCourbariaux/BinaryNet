
import time

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
    
def gemm(gemm_kernel,A,B,C,A_rows,A_cols,B_cols):

    # dimensions
    assert A_cols%16 == 0 # Block size

    # Launching GEMM GPU kernel            
    block_size = 16
    block = (block_size,block_size,1)
    grid = (B_cols / block_size+1, A_rows / block_size+1) # better too many blocks than too little
    gemm_kernel(A,B,C, np.intc(A_rows), np.intc(A_cols), np.intc(B_cols), block= block, grid=grid)
    
def concatenation_rows(concatenate_rows_kernel,A,A_conc,A_rows,A_cols):
    
    assert A_cols%32 == 0 # concatenation step
    
    block_size = 64            
    block = (block_size,1,1)
    grid = (A_rows*A_cols/(block_size*32)+1,1)
    concatenate_rows_kernel(A,A_conc, np.intc(A_rows*A_cols/32), block= block, grid=grid)
    
def concatenation_cols(concatenate_cols_kernel,A,A_conc,A_rows,A_cols):
    
    assert A_rows%32 == 0 # concatenation step
    
    block_size = 64 
    block = (block_size,1,1)
    grid = (A_cols/block_size+1,1)
    concatenate_cols_kernel(A,A_conc, np.intc(A_rows), np.intc(A_cols), block= block, grid=grid)
    
    
def sign(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)
    
if __name__ == "__main__":   
    
    context = pycuda.autoinit.context
    
    print "Building the kernels..."
    
    mod = SourceModule(open("binary_kernels.cu").read())
    gemm_kernel = mod.get_function("gemm")
    concatenate_rows_kernel = mod.get_function("concatenate_rows_kernel")
    concatenate_cols_kernel = mod.get_function("concatenate_cols_kernel")
    xnor_gemm_kernel = mod.get_function("xnor_gemm")
    
    print "Loading matrices to device..."
    
    # Matrices dimensions
    N = 8192
    A_rows = N
    A_cols = N
    B_cols = N
    # A_rows = 784
    # A_cols = 1024
    # B_cols = 4096
    
    # A is a matrix randomly filled with 1 and -1
    A = sign(np.random.randn(A_rows,A_cols))
    A = A.astype(np.float32)
    A_gpu = cuda.mem_alloc(A.nbytes)
    cuda.memcpy_htod(A_gpu, A)
    
    # B is a matrix randomly filled with 1 and -1
    B = sign(np.random.randn(A_cols,B_cols))
    B = B.astype(np.float32)
    B_gpu = cuda.mem_alloc(B.nbytes)
    cuda.memcpy_htod(B_gpu, B)
    
    # C is the resulting matrix
    C1 = np.zeros((A_rows,B_cols)).astype(np.float32)
    C2 = np.zeros((A_rows,B_cols)).astype(np.float32)
    C_gpu = cuda.mem_alloc(C1.nbytes)
    
    print "XNOR kernel..."
    
    # wait until the GPU is done with the work
    context.synchronize()
    # kernel timing
    start_time = time.time()
    
    # concatenate A
    A_conc = cuda.mem_alloc(A.nbytes/32)
    concatenation_rows(concatenate_rows_kernel,A_gpu,A_conc,A_rows,A_cols)
    # concatenate B
    B_conc = cuda.mem_alloc(B.nbytes/32)
    concatenation_cols(concatenate_cols_kernel,B_gpu,B_conc,A_cols,B_cols)
    # XNOR GEMM
    gemm(xnor_gemm_kernel,A_conc,B_conc,C_gpu,A_rows,A_cols/32,B_cols)
    # Free concatenated memory
    A_conc.free()
    B_conc.free()
    
    # wait until the GPU is done with the work
    context.synchronize()
    # kernel timing
    execution_time = time.time() - start_time
    print(" execution_time = "+str(execution_time)+"s")
    
    # get the result
    cuda.memcpy_dtoh(C2,C_gpu)
    
    print "Baseline kernel..."
    
    # wait until the GPU is done with the work
    context.synchronize()
    # kernel timing
    start_time = time.time()

    gemm(gemm_kernel,A_gpu,B_gpu,C_gpu,A_rows,A_cols,B_cols)
    
    # wait until the GPU is done with the work
    context.synchronize()
    # kernel timing
    execution_time = time.time() - start_time
    print(" execution_time = "+str(execution_time)+"s")
    
    # get the result
    cuda.memcpy_dtoh(C1,C_gpu)
    
    print "Comparing the results..."
    
    print " np.allclose(C1, C2) = " + str(np.allclose(C1, C2))
