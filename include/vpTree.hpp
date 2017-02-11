#pragma once

#include "../include/helper.hpp"
#include "cuda_runtime.h"
#include <vector>
#include <stack>
#include <algorithm>
#include <stdio.h>

#define DIM 2

namespace cu_vp
{
	/**
	 * Structure used to define the VP tree on the GPU.
	 */
	struct CUDA_VPNode
	{
		double threshold;
		int index;
		int left, right;
		int padding; // Not needed. Can be replaced

		CUDA_VPNode() : threshold(0.), index(0), left(-1), right(-1) {}
	};


	/**
	 * Structure defining a point in the VP tree
	 */
	struct Point
	{
		double coords[DIM];
	};

	typedef double(*DistFunc)(const Point& a, const Point& b);

	/**
	 * Default distance function to use to create the VP tree (CPU side). Can be any distance function
	 * as long as it abides triangle inequality
	 */
	double euclidean_distance(const Point& a, const Point& b);

	/**
	 * Default distance function to use to create the VP tree (GPU side). Can be any distance function
	 * as long as it abides triangle inequality.
	 */
	__device__ double gpu_euclidean_distance(const Point& a, const Point& b);



	/**
	 * Vantage Point tree that performs batches of queries in parallel on CUDA devices.
	 * Can perform k-nearest neighbour search as well as fixed radius searches.
	 * Construction of the tree is performed on the CPU.
	 * Note: due to interactions between templates and .cu files, I haven't been able to make
	 * this a templated class, which means the Point datatype is restricted at compile time---meaning
	 * only a single type of point can exist per compile. i.e Can't have 2D and 3D points in same executable
	 * One way around this is to use 3D points in both cases, but define a 2D distance function which only uses
	 * the first two coordinates.
	 */
	class CUDA_VPTree
	{
	public:
		CUDA_VPTree();
		~CUDA_VPTree();

		/**
		 * Instantiates the VP tree using data set
		 * \param data - Points to construct VP tree from
		 */
		void createVPTree(std::vector<Point> &data);

		/**
		 * Performs a batched k nearest neighbour search.
		 * \param queries - Points to find k nearest neighbours to
		 * \param k - Num neighbours to find per query point
		 * \param[out] indices - Indices of the k nearest neighbours to each query point
		 * \param[out] distances - Distances to the k nearest neighbours to each query point
		 */
		void knnSearch(const std::vector<Point> &queries, const int k,
					   std::vector<int> &indices, std::vector<double> &distances);

		/**
		* Performs a batched fixed radius search.
		* \param queries - Points to find all neighbours within radius
		* \param fr - Radius to use
		* \param[out] count - Number of points with range of each query
		*/
		void frSearch(const std::vector<Point> &queries, const double fr,
					  std::vector<int> &count);

		/**
		 * Replace the default euclidean distance functions with some other metric
		 * \param newDistanceFunc - Distance function to be used on host (creation of VP Tree)
		 * \param newGpuDistanceFunc - Distance function to be used on the device (searches). Function needs a
		 *								__device__ attribute attached
		 */
		void injectDistanceFunc(DistFunc newDistanceFunc,
								/*__device__*/ DistFunc newGpuDistanceFunc);
	private:
		/**
		 * Builds the VP tree
		 * \param data - Data points to use
		 */
		void buildFromPoints(std::vector<Point> &data);

		/**
		 * \todo - Rearrange this so can do streamed VP searches. i.e have a cpu_nodes to
		 * store all points, and use gpu_nodes as the staging buffer on the GPU. Likewise for
		 * gpu_points
		 */

		 /** Node array storing the VP tree */
		CUDA_VPNode *gpu_nodes;
		/** Data points stored on the GPU */
		Point *gpu_points;

		/** Number of points in tree */
		size_t num_points;

		/** Flag indicating whether VP tree is in valid state for searching */
		bool tree_valid;

		/** Distance functions */
		DistFunc distanceFunc;
		DistFunc gpuDistanceFunc;

		struct DistComparator
		{
			const Point& item;
			double(*compDistanceFunc)(const Point& a, const Point& b);
			DistComparator(const Point& item, double(*newDistanceFunc)(const Point& a, const Point& b)) : item(item), compDistanceFunc(newDistanceFunc) {}
			bool operator()(const Point& a, const Point& b) {
				return compDistanceFunc(item, a) < compDistanceFunc(item, b);
			}
		};
	};
}