#include "../include/helper.hpp"

void CheckCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}