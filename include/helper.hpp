#pragma once

#include "cuda_runtime.h"
#include <iostream>


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

void CheckCUDAError(const char *msg);