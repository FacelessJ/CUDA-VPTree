#pragma once

#include "cuda_runtime.h"
#include <iostream>


/**
 * Macro to wrap around all cuda* calls to check for errors
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s (%d) %s %d\n", cudaGetErrorString(code), code, file, line);
		system("pause");
		if(abort) exit(code);
	}
}

/**
 * Checks for any error currently existing, typically after running kernel
 * \param msg - Text string to add to output if there is an error
 */
void CheckCUDAError(const char *msg);

/**
* Reports current usage of device memory
* \param msg - Text string tag to add to output
*/
void reportDeviceMemUsage(const char *msg);
