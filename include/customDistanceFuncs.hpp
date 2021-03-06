#pragma once
#include "../include/vpTree.hpp"

/**
 * Sample file showing how to define custom distance functions for use
 * with CUDA_VPTree.
 * 
 * For regular metrics, such as manhattan or maxnorm, simple functions
 * will do.
 *
 * However, if the metric requires extra data (for instance, using a
 * wrapped euclidean distance on a square area, thus the dist func
 * needing to know the width/height of the space), you can use a static
 * class to encapsulate the extra data, such that it doesn't pollute the
 * global namespace
 */

class WrappedDistance {
public:
	/** Custom data required by distance function */
	static double wrap;

	/** Host side custom distance function */
	static double distanceWrap(const cu_vp::Point& a, const cu_vp::Point& b);

	/** Device side custom distance function */
	static __device__ double gpuDistanceWrap(const cu_vp::Point& a, const cu_vp::Point& b);

	/** Get host side pointer to device function */
	static cu_vp::DistFunc getFuncPtr();
};

