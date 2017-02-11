#include "../include/helper.hpp"

void CheckCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void reportDeviceMemUsage(const char *msg)
{
	size_t free, total;
	gpuErrchk(cudaMemGetInfo(&free, &total));
	printf("%s: %zd KB free of %zd\n", msg, free / 1024, total / 1024);

}
