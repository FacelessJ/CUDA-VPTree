#include "cuda_runtime.h"
#include "../include/customDistanceFuncs.hpp"
#include <stdio.h>

double WrappedDistance::wrap = 1;

double WrappedDistance::distanceWrap(const cu_vp::Point& a, const cu_vp::Point& b)
{
	double total = 0.;
	for(size_t i = 0; i < DIM; ++i) {
		total = (b.coords[i] - a.coords[i]) * (b.coords[i] - a.coords[i]);
	}
	return sqrt(total);
}

__device__ double WrappedDistance::gpuDistanceWrap(const cu_vp::Point& a, const cu_vp::Point& b)
{
	double total = 0.;
	for(size_t i = 0; i < DIM; ++i) {
		total = (b.coords[i] - a.coords[i]) * (b.coords[i] - a.coords[i]);
	}
	printf("Total found: %f\n", total);
	return sqrt(total);
}

__device__ cu_vp::DistFunc func = WrappedDistance::gpuDistanceWrap;

cu_vp::DistFunc WrappedDistance::getFuncPtr()
{
	cu_vp::DistFunc ret;
	gpuErrchk(cudaMemcpyFromSymbol(&ret, func, sizeof(cu_vp::DistFunc)));
	return ret;
}


