#include "cuda_runtime.h"
#include "../include/helper.hpp"
#include "../include/vpTree.hpp"

#include <algorithm>
#include <stack>
#include <stdio.h>

// Constant stack size for searching.
// Needs to be at least ceil(log2(N)) + 1, where N is number of data points
// in the VP Tree.
// i.e a stack size of 32 will handle 2^31 data points
#define CUDA_STACK_SIZE 25

CUDA_VPTree::CUDA_VPTree() : tree_valid(false)
{
}

CUDA_VPTree::~CUDA_VPTree()
{
	gpuErrchk(cudaFree(gpu_nodes));
	//gpuErrchk(cudaFree(gpu_indices));
	gpuErrchk(cudaFree(gpu_points));
}

double distance(const Point& a, const Point& b) {
	double total = 0.;
	for(size_t i = 0; i < DIM; ++i) {
		total = (b.coords[i] - a.coords[i]) * (b.coords[i] - a.coords[i]);
	}
	return sqrt(total);
}

__device__ double gpu_distance(const Point& a, const Point& b) {
	double total = 0.;
	for(size_t i = 0; i < DIM; ++i) {
		total = (b.coords[i] - a.coords[i]) * (b.coords[i] - a.coords[i]);
	}
	return sqrt(total);
}

void CUDA_VPTree::createVPTree(std::vector<Point> &data)
{
	num_points = data.size();

	
	buildFromPoints(data);
	gpuErrchk(cudaMalloc((void**)&gpu_points, sizeof(Point)*num_points));
	
	size_t free, total;
	gpuErrchk(cudaMemGetInfo(&free, &total));
	printf("Create: Alloc'd %zd KB. %zd KB free\n", (sizeof(Point)*num_points) / 1024, free / 1024);
	gpuErrchk(cudaMemcpy(gpu_points, &(data[0]), sizeof(Point)*num_points, cudaMemcpyHostToDevice));
	tree_valid = true;
}

__device__ void Search(const CUDA_VPNode *nodes,/*const int *indexes,*/ const Point *pts, const Point &query, int *ret_index, double *ret_dist)
{
	// Find the first closest node, this will be the upper bound for the next searches
	//int best_node = 0;
	int best_idx = -1;
	double best_dist = DBL_MAX;
	//float radius = 0;

	int nodeStack[CUDA_STACK_SIZE];
	nodeStack[0] = -1;
	int stackPtr = 0;
	int currNodeIdx = 0; //Start at root
	double tau = DBL_MAX;
	while(stackPtr >= 0 || currNodeIdx != -1) {
		if(currNodeIdx != -1) {
			double dist = gpu_distance(query, pts[currNodeIdx]);

			if(dist < tau) {
				//best_node = currNodeIdx;
				best_idx = currNodeIdx;
				best_dist = dist;
				tau = dist;
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
					best_idx = -1;
					best_dist = -1.0;
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
					best_idx = -1;
					best_dist = -1.0;
					break;
				}
			}
		}
		if(stackPtr >= 0) {
			currNodeIdx = nodeStack[stackPtr--];
		}
	}
	*ret_index = best_idx;
	*ret_dist = best_dist;
}

__global__ void SearchBatch(const CUDA_VPNode *nodes, /*const int *indexes,*/ const Point *pts, int num_pts, Point *queries, int num_queries, int *ret_index, double *ret_dist)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= num_queries)
		return;

	Search(nodes, /*indexes,*/ pts, queries[idx], &ret_index[idx], &ret_dist[idx]);
	//printf("Query %d (%f, %f) had closest point at %d, %f\n", idx, queries[idx].coords[0], queries[idx].coords[1], ret_index[idx], ret_dist[idx]);
}


void CUDA_VPTree::search(const std::vector<Point>& queries, std::vector<int>& indices, std::vector<double>& distances)
{
	if(!tree_valid)
		return;
	int blockSize = 512;
	int numBlocks = ((int)queries.size() + blockSize - 1) / blockSize;

	Point *gpu_queries;
	int *gpu_ret_indices;
	double *gpu_ret_dists;

	indices.resize(queries.size());
	distances.resize(queries.size());

	gpuErrchk(cudaMalloc((void**)&gpu_queries, sizeof(Point)*queries.size()));
	size_t free, total;
	gpuErrchk(cudaMemGetInfo(&free, &total));
	printf("Queries: Alloc'd %zd KB. %zd KB free\n", sizeof(Point)*queries.size() / 1024, free / 1024);

	gpuErrchk(cudaMalloc((void**)&gpu_ret_indices, sizeof(int)*queries.size()));
	gpuErrchk(cudaMemGetInfo(&free, &total));
	printf("Indices: Alloc'd %zd KB. %zd KB free\n", sizeof(int)*queries.size() / 1024, free / 1024);


	gpuErrchk(cudaMalloc((void**)&gpu_ret_dists, sizeof(double)*queries.size()));
	gpuErrchk(cudaMemGetInfo(&free, &total));
	printf("Dists: Alloc'd %zd KB. %zd KB free\n", sizeof(double)*queries.size() / 1024, free / 1024);


	gpuErrchk(cudaMemcpy(gpu_queries, &queries[0], sizeof(Point)*queries.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaThreadSynchronize());
	CheckCUDAError("Pre Batch search");

	printf("Searching on GPU with %d blocks with %d threads per block\n", numBlocks, blockSize);

	SearchBatch<<<numBlocks, blockSize>>>(gpu_nodes, gpu_points, (int)num_points, gpu_queries, (int)queries.size(), gpu_ret_indices, gpu_ret_dists);
	CheckCUDAError("Batch search");
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaThreadSynchronize());

	gpuErrchk(cudaMemcpy(&indices[0], gpu_ret_indices, sizeof(int)*queries.size(), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&distances[0], gpu_ret_dists, sizeof(double)*queries.size(), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(gpu_queries));
	gpuErrchk(cudaFree(gpu_ret_indices));
	gpuErrchk(cudaFree(gpu_ret_dists));
}

int CUDA_VPTree::treeDepth()
{
	return (int)ceil(log2(num_points));
}

void CUDA_VPTree::buildFromPoints(std::vector<Point> &data)
{
	int nodeCount = 0;
	typedef std::pair<int, int> Range;
	std::vector<CUDA_VPNode> cpu_nodes(num_points);
	//std::vector<int> cpu_indices(num_points);
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
				DistComparator(data[lower]));

			// what was the median?
			node->threshold = distance(data[lower], data[median]);

			node->index = lower;
			node->left = median != (lower + 1) ? lower + 1 : -1;
			node->right = upper != median ? median : -1;
			ranges_to_process.push(Range(median, upper));
			ranges_to_process.push(Range(lower + 1, median));
		}
	}

	gpuErrchk(cudaMalloc((void**)&gpu_nodes, sizeof(CUDA_VPNode)*num_points));
	size_t free, total;
	gpuErrchk(cudaMemGetInfo(&free, &total));
	printf("Build: Alloc'd %zd KB. %zd KB free\n", sizeof(CUDA_VPNode)*num_points / 1024, free / 1024);

	//gpuErrchk(cudaMalloc((void**)&gpu_indices, sizeof(int)*num_points));

	/*FILE* fp = fopen("gpu_tree.txt", "w");
	for(size_t i = 0; i < cpu_nodes.size(); ++i) {
		fprintf(fp, "%d: %d %d: %f\n", cpu_nodes[i].index, cpu_nodes[i].left, cpu_nodes[i].right, cpu_nodes[i].threshold);
	}
	fclose(fp);*/
	gpuErrchk(cudaMemcpy(gpu_nodes, &(cpu_nodes[0]), sizeof(CUDA_VPNode)*num_points, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(gpu_indices, &(cpu_indices[0]), sizeof(int)*num_points, cudaMemcpyHostToDevice));
}

