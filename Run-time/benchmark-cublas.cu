
#include "binary_kernels.cu"
#include <iostream>
#include <chrono>
#include <cublas_v2.h>
using namespace std;
// to run: nvcc benchmark-cublas.cu -std=c++11 -lcublas && ./a.out
int main() {
	int N = 8192;

	// prepare data
	float *A = (float*)malloc(N * N * sizeof(float));
	float *B = (float*)malloc(N * N * sizeof(float));
	for (int i = 0; i < N * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		A[i] = (x > 0.5) ? 1 : -1;
		x = rand() / RAND_MAX;
		B[i] = (x > 0.5) ? 1 : -1;
	}

	// copy to cuda
	float *fA, *fB, *fC;
	cudaMalloc(&fA, N * N * sizeof(float));
	cudaMalloc(&fB, N * N * sizeof(float));
	cudaMalloc(&fC, N * N * sizeof(float));
	cudaMemcpy(fA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

	auto test_xnor = [&]() {
		unsigned int *Aconc, *Bconc;
		cudaMalloc(&Aconc, N * N);
		cudaMalloc(&Bconc, N * N);
		cudaMemset(fC, 0, N * N * sizeof(int));

		auto start = chrono::high_resolution_clock::now();
		int block = 64, grid = N * N / (block * 32)  + 1;
		concatenate_rows_kernel<<<grid, block>>>(fA, Aconc, N * N / 32);

		grid = N / block + 1;
		concatenate_cols_kernel<<<grid, block>>>(fB, Bconc, N, N);
		cudaDeviceSynchronize();

		dim3 blockDim(16, 16);
		dim3 gridDim(N / 16 + 1, N / 16 + 1);
		xnor_gemm<<<gridDim, blockDim>>>(Aconc, Bconc, fC, N, N / 32, N);
		cudaDeviceSynchronize();

		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		cout << "XNOR GEMM kernel time: " << diff.count() << " s\n";

		float* result = (float*)malloc(N * N * sizeof(float));
		cudaMemcpy(result, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_xnor = test_xnor();


	auto test_gemm = [&]() {
		cudaMemset(fC, 0, N * N * sizeof(int));
		dim3 blockDim(16, 16);
		dim3 gridDim(N / 16 + 1, N / 16 + 1);
		auto start = chrono::high_resolution_clock::now();
		gemm<<<gridDim, blockDim>>>(fA, fB, fC, N, N, N);
		cudaDeviceSynchronize();
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		cout << "GEMM kernel time: " << diff.count() << " s\n";

		float* result = (float*)malloc(N * N * sizeof(float));
		cudaMemcpy(result, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_gemm = test_gemm();


	auto test_cublas = [&]() {
		cudaMemset(fC, 0, N * N * sizeof(int));
		cublasHandle_t handle;
		cublasCreate(&handle);

		auto start = chrono::high_resolution_clock::now();
		float alpha = 1.0, beta = 0.0;
		// cublas use column-major
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, fB, N, fA, N, &beta, fC, N);
		cudaDeviceSynchronize();
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		cout << "cublas time: " << diff.count() << " s\n";

		float* result = (float*)malloc(N * N * sizeof(float));
		cudaMemcpy(result, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_cublas = test_cublas();

	auto check_result = [&](float* p1, float* p2) {
		for (int i = 0; i < N * N; i ++) {
			float diff = p1[i] - p2[i];
			if (fabs(diff) > 1e-6) {
				printf("%f\n", diff);
				return false;
			}
		}
		return true;
	};

	printf("success: %d\n", check_result(result_gemm, result_xnor));
	printf("success: %d\n", check_result(result_gemm, result_cublas));

}
