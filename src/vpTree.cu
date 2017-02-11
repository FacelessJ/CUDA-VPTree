#pragma once

#include "cuda_runtime.h"
#include "../include/helper.hpp"
#include "../include/vpTree.hpp"

#include <stdio.h>

// Constant stack size for searching.
// Needs to be at least ceil(log2(N)) + 1, where N is number of data points
// in the VP Tree.
// i.e a stack size of 32 will handle 2^31 data points
#define CUDA_STACK_SIZE 32

namespace cu_vp
{

	double euclidean_distance(const Point& a, const Point& b) {
		double total = 0.;
		for(size_t i = 0; i < DIM; ++i) {
			total = (b.coords[i] - a.coords[i]) * (b.coords[i] - a.coords[i]);
		}
		return sqrt(total);
	}

	__device__ double gpu_euclidean_distance(const Point& a, const Point& b) {
		double total = 0.;
		for(size_t i = 0; i < DIM; ++i) {
			total = (b.coords[i] - a.coords[i]) * (b.coords[i] - a.coords[i]);
		}
		return sqrt(total);
	}

	__device__ DistFunc g_gpuDistanceFunc = gpu_euclidean_distance;

	/**
	 * Performs a knn search for a single point
	 * \param nodes - Pointer to root node of the tree
	 * \param pts - Data points, mapped to nodes
	 * \param query - Query point
	 * \param k - How many neighbours to find
	 * \param[out] ret_index - Indices of nearest neighbours
	 * \param[out] ret_dist - Distances to nearest neighbours
	 * \param distFunc - Distance function to use to compare points
	 */
	__device__ void KNNSearch(const CUDA_VPNode *nodes, const Point *pts,
							  const Point &query, const int k, int *ret_index,
							  double *ret_dist, DistFunc distFunc)
	{
		/** Stack for traversing the tree */
		int nodeStack[CUDA_STACK_SIZE];
		nodeStack[0] = -1;
		int stackPtr = 0;

		for(int i = 0; i < k; ++i) {
			ret_dist[i] = DBL_MAX;
		}


		int currNodeIdx = 0; //Start at root
		double tau = DBL_MAX;
		while(stackPtr >= 0 || currNodeIdx != -1) {
			if(currNodeIdx != -1) {
				double dist = distFunc(query, pts[currNodeIdx]);

				if(dist < tau) {
					if(k == 1) {
						ret_dist[0] = dist;
						ret_index[0] = currNodeIdx;
					}
					else {
						//insert at right position
						for(int i = k - 2; i >= 0; --i) {
							if(dist < ret_dist[i]) {
								ret_dist[i + 1] = ret_dist[i];
								ret_index[i + 1] = ret_index[i];
								if(i == 0) {
									ret_dist[i] = dist;
									ret_index[i] = currNodeIdx;
								}
							}
							else {
								ret_dist[i + 1] = dist;
								ret_index[i + 1] = currNodeIdx;
								break;
							}
						}
					}
					//Set tau limit. Since ret_dist is instantiated with
					//DBL_MAX at all values, we don't need to check if
					//list is full
					tau = ret_dist[k-1]; 

				}

				if(nodes[currNodeIdx].left == -1 && nodes[currNodeIdx].right == -1) {
					currNodeIdx = -1;
					continue;
				}

				if(dist < nodes[currNodeIdx].threshold) {
					if(dist + tau >= nodes[currNodeIdx].threshold) {
						nodeStack[++stackPtr] = nodes[currNodeIdx].right;
					}
					if(dist - tau <= nodes[currNodeIdx].threshold) {
						nodeStack[++stackPtr] = nodes[currNodeIdx].left;
					}
					if(stackPtr > CUDA_STACK_SIZE)
					{
						printf("ERROR: stackPtr larger than stack size!\n");
						break;
					}
				}
				else {
					if(dist - tau <= nodes[currNodeIdx].threshold) {
						nodeStack[++stackPtr] = nodes[currNodeIdx].left;
					}
					if(dist + tau >= nodes[currNodeIdx].threshold) {
						nodeStack[++stackPtr] = nodes[currNodeIdx].right;
					}
					if(stackPtr > CUDA_STACK_SIZE)
					{
						printf("ERROR: stackPtr larger than stack size!\n");
						break;
					}
				}
			}
			if(stackPtr >= 0) {
				currNodeIdx = nodeStack[stackPtr--];
			}
		}
	}

	/**
	 * Kernel distributing knn search jobs
	 * \param nodes - Pointer to root node of the tree
	 * \param pts - Data points, mapped to nodes
	 * \param num_pts - Number of points
	 * \param query - Query point
	 * \param num_queries - Number of queries
	 * \param k - Number of neighbours to find for each point
	 * \param[out] ret_index - Indices of nearest neighbours
	 * \param[out] ret_dist - Distances to nearest neighbours
	 * \param distFunc - Distance function to use to compare points
	 */
	__global__ void KNNSearchBatch(const CUDA_VPNode *nodes, const Point *pts, int num_pts,
								   Point *queries, int num_queries, const int k, int *ret_index,
								   double *ret_dist, DistFunc distFunc)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;

		if(idx >= num_queries)
			return;

		KNNSearch(nodes, pts, queries[idx], k, &ret_index[idx * k], &ret_dist[idx * k], distFunc);
	}

	CUDA_VPTree::CUDA_VPTree() : gpu_nodes(nullptr), gpu_points(nullptr), num_points(0),
		tree_valid(false), distanceFunc(&euclidean_distance), gpuDistanceFunc(nullptr)
	{
		gpuErrchk(cudaMemcpyFromSymbol(&gpuDistanceFunc, g_gpuDistanceFunc, sizeof(DistFunc)));
	}

	CUDA_VPTree::~CUDA_VPTree()
	{
		gpuErrchk(cudaFree(gpu_nodes));
		gpuErrchk(cudaFree(gpu_points));
	}

	void CUDA_VPTree::injectDistanceFunc(DistFunc newDistanceFunc,
										 DistFunc newGpuDistanceFunc)
	{
		distanceFunc = newDistanceFunc;
		gpuDistanceFunc = newGpuDistanceFunc;
	}

	void CUDA_VPTree::createVPTree(std::vector<Point> &data)
	{
		num_points = data.size();

		buildFromPoints(data);

		if(gpu_points != nullptr) {
			gpuErrchk(cudaFree(gpu_points));
			gpu_points = nullptr;
		}
		gpuErrchk(cudaMalloc((void**)&gpu_points, sizeof(Point)*num_points));

		reportDeviceMemUsage("Create");
		gpuErrchk(cudaMemcpy(gpu_points, &(data[0]), sizeof(Point)*num_points, cudaMemcpyHostToDevice));
		tree_valid = true;
	}

	void CUDA_VPTree::knnSearch(const std::vector<Point> &queries, const int k, std::vector<int> &indices, std::vector<double> &distances)
	{
		if(!tree_valid)
			return;
		int blockSize = 512;
		int numBlocks = ((int)queries.size() + blockSize - 1) / blockSize;

		Point *gpu_queries;
		int *gpu_ret_indices;
		double *gpu_ret_dists;

		indices.resize(queries.size() * k);
		distances.resize(queries.size() * k);

		gpuErrchk(cudaMalloc((void**)&gpu_queries, sizeof(Point)*queries.size()));
		reportDeviceMemUsage("Queries");

		gpuErrchk(cudaMalloc((void**)&gpu_ret_indices, sizeof(int)*indices.size()));
		reportDeviceMemUsage("Indices");

		gpuErrchk(cudaMalloc((void**)&gpu_ret_dists, sizeof(double)*distances.size()));
		reportDeviceMemUsage("Dists");

		gpuErrchk(cudaMemcpy(gpu_queries, &queries[0], sizeof(Point)*queries.size(), cudaMemcpyHostToDevice));
		gpuErrchk(cudaThreadSynchronize());
		CheckCUDAError("Pre Batch search");

		printf("Searching on GPU with %d blocks with %d threads per block\n", numBlocks, blockSize);

		KNNSearchBatch << <numBlocks, blockSize >> > (gpu_nodes, gpu_points, (int)num_points,
													  gpu_queries, (int)queries.size(), k,
													  gpu_ret_indices, gpu_ret_dists, 
													  gpuDistanceFunc);
		CheckCUDAError("Batch search");
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaThreadSynchronize());

		gpuErrchk(cudaMemcpy(&indices[0], gpu_ret_indices, sizeof(int)*indices.size(), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&distances[0], gpu_ret_dists, sizeof(double)*distances.size(), cudaMemcpyDeviceToHost));

		gpuErrchk(cudaFree(gpu_queries));
		gpuErrchk(cudaFree(gpu_ret_indices));
		gpuErrchk(cudaFree(gpu_ret_dists));
	}

	void CUDA_VPTree::frSearch(const std::vector<Point>& queries, const double fr, std::vector<int>& count)
	{
		/** \todo Implement this */
		/** Need to consider how to deal with dynamic number of points being returned.
		    Going to not return the individual distances to each point, nor the indices,
			but instead the count, and provide a function/user data pointer to apply
			operation to each point within threshold*/
		/** Possibly have another version with const std::vector<double> fr which defines
		 * a separate fr threshold for each query */
	}


	void CUDA_VPTree::buildFromPoints(std::vector<Point> &data)
	{
		int nodeCount = 0;
		typedef std::pair<int, int> Range;
		std::vector<CUDA_VPNode> cpu_nodes(num_points);
		std::stack<Range> ranges_to_process;
		ranges_to_process.push(Range(0, (int)num_points));

		while(ranges_to_process.empty() == false) {

			int upper = ranges_to_process.top().second, lower = ranges_to_process.top().first;
			ranges_to_process.pop();
			if(lower == upper) {
				continue;
			}

			CUDA_VPNode* node = &(cpu_nodes[nodeCount++]);
			node->index = lower;

			if(upper - lower > 1) {

				// choose an arbitrary point and move it to the start
				double m = (double)rand() / RAND_MAX;
				m = 0.5;
				int i = (int)(m * (upper - lower - 1)) + lower;
				std::swap(data[lower], data[i]);

				int median = (upper + lower) / 2;

				// partitian around the median distance
				std::nth_element(
					data.begin() + lower + 1,
					data.begin() + median,
					data.begin() + upper,
					DistComparator(data[lower], distanceFunc));

				// what was the median?
				node->threshold = distanceFunc(data[lower], data[median]);

				node->index = lower;
				node->left = median != (lower + 1) ? lower + 1 : -1;
				node->right = upper != median ? median : -1;
				ranges_to_process.push(Range(median, upper));
				ranges_to_process.push(Range(lower + 1, median));
			}
		}

		if(gpu_nodes != nullptr) {
			gpuErrchk(cudaFree(gpu_nodes));
			gpu_nodes = nullptr;
		}
		gpuErrchk(cudaMalloc((void**)&gpu_nodes, sizeof(CUDA_VPNode)*num_points));
		size_t free, total;
		gpuErrchk(cudaMemGetInfo(&free, &total));
		printf("Build: Alloc'd %zd KB. %zd KB free\n", sizeof(CUDA_VPNode)*num_points / 1024, free / 1024);

		gpuErrchk(cudaMemcpy(gpu_nodes, &(cpu_nodes[0]), sizeof(CUDA_VPNode)*num_points, cudaMemcpyHostToDevice));
	}
}