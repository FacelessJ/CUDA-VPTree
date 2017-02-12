#include "../include/vpTree.hpp"
#include "../include/cpu_vpTree.hpp"
#include "../include/helper.hpp"
#include "../include/customDistanceFuncs.hpp"
#include "cuda_runtime.h"

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>
#include <chrono>
#include <iostream>

int main()
{
	typedef std::chrono::high_resolution_clock Clock;
	
	//todo: Memory management: For ultra large trees (i.e host can hold but device cannot)
	//		implement a step down algorithm, which processes the VP tree in halves (or quarters, etc)
	//		whereby:
	//		Let T = whole tree
	//		Let S be stack of trees to process
	//		Let Tau[num_queries] = infinity be the progress for each query. Might be able to ignore this and just use dist.back() (or infinity if dist.size() < k)
	//		Let res_dist[num_queries] and res_ids[num_queries] be the accumulated result lists for each query
	//		while(!stack.emtpy())
	//			if(sizeof(S.head()) > device_mem)
	//				Let T1 and T2 be near/far subtrees of T (ordered based on which to process first)
	//				Push T1 and T2 onto stack
	//			else
	//				Let P = S.head(); S.pop()
	//				Process P as per normal, except pass in Tau, res_dist and res_ids to work from
	//				Update Tau, res_dist and res_id as necessary
	//
	//		With this process, each segment of the tree is only loaded onto the device once
	//		However, with large query sets, we might need to reproduce this process for segments of the query set Q
	//		i.e break Q into two halves, and repeat above for each half

	size_t free, total;

	gpuErrchk(cudaMemGetInfo(&free, &total));

	printf("%zd KB free of total %zd KB\n", free / 1024, total / 1024);

	int start = 100;
	int end = 100;
	FILE* timings = fopen("timings.csv", "w");
	if(timings)
		fprintf(timings, "Num Points,Num Queries,GPU create,CPU create,GPU search,CPU search\n");
	else
		printf("Cannot open timings.csv for write access. Will continue without recording timing data\n");
	for(int dataSize = start; dataSize <= end; dataSize *= 10) {
		for(int querySize = start; querySize <= end; querySize *= 10) {
			printf("\n**********************\n");
			printf("Performing %d queries on %d data points\n", querySize, dataSize);
			cu_vp::CUDA_VPTree GPU_tree;
			GPU_tree.injectDistanceFunc(&cu_vp::euclidean_distance, WrappedDistance::getFuncPtr());
			VpTree<cu_vp::Point, VDistance<cu_vp::Point> > CPU_tree;

			std::vector<cu_vp::Point> data(dataSize);
			std::vector<cu_vp::Point> queries(querySize);

			if(timings)
				fprintf(timings, "%d,%d,", dataSize, querySize);

			for(size_t i = 0; i < data.size(); ++i) {
				data[i].coords[0] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
				data[i].coords[1] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
			}

			for(size_t i = 0; i < queries.size(); i++) {
				queries[i].coords[0] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
				queries[i].coords[1] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
			}

			std::vector<cu_vp::Point> dataCpy(data.size());
			memcpy(&(dataCpy[0]), &(data[0]), sizeof(cu_vp::Point)*data.size());

			printf("Creating GPU Tree\n");
			auto gpu_create_t1 = Clock::now();
			GPU_tree.createVPTree(data);
			auto gpu_create_t2 = Clock::now();
			auto gpu_create_time_span = std::chrono::duration_cast<std::chrono::duration<double>>(gpu_create_t2 - gpu_create_t1);
			if(timings)
				fprintf(timings, "%f,", gpu_create_time_span.count());
			printf("GPU Tree created\n");

			printf("Creating CPU Tree\n");
			auto cpu_create_t1 = Clock::now();
			CPU_tree.create(dataCpy);
			auto cpu_create_t2 = Clock::now();
			auto cpu_create_time_span = std::chrono::duration_cast<std::chrono::duration<double>>(cpu_create_t2 - cpu_create_t1);
			if(timings)
				fprintf(timings, "%f,", cpu_create_time_span.count());
			printf("CPU Tree created\n");

			std::vector<int> gpu_indices;
			std::vector<double> gpu_dists;
			int k = 1;
			printf("Searching GPU Tree\n");
			auto gpu_t1 = Clock::now();
			GPU_tree.knnSearch(queries, k, gpu_indices, gpu_dists);
			auto gpu_t2 = Clock::now();
			std::chrono::duration<double> gpu_time_span = std::chrono::duration_cast<std::chrono::duration<double>>(gpu_t2 - gpu_t1);
			if(timings)
				fprintf(timings, "%f,", gpu_time_span.count());
			printf("GPU Tree searched\n");

			std::vector<int> gpu_counts;
			GPU_tree.frSearch(queries, gpu_dists, gpu_counts);
			/*for(size_t i = 0; i < gpu_counts.size(); ++i) {
				printf("%d : %d\n", i, gpu_counts[i]);
			}*/

			printf("Searching CPU Tree\n");
			auto cpu_t1 = Clock::now();
			std::vector<cu_vp::Point> cpu_results;
			std::vector<double> cpu_dists;
			std::vector<cu_vp::Point> out_res;
			std::vector<double> out_dist;
			for(size_t i = 0; i < queries.size(); ++i) {
				CPU_tree.search(queries[i], k, &out_res, &out_dist);
				cpu_results.insert(cpu_results.end(), out_res.begin(), out_res.end());
				cpu_dists.insert(cpu_dists.end(), out_dist.begin(), out_dist.end());
			}
			auto cpu_t2 = Clock::now();
			std::chrono::duration<double> cpu_time_span = std::chrono::duration_cast<std::chrono::duration<double>>(cpu_t2 - cpu_t1);
			if(timings)
				fprintf(timings, "%f\n", cpu_time_span.count());
			printf("CPU Tree searched\n");

			//Verify results
			/*bool equal = true;
			double threshold = 0.000001;
			printf("gpu_dists.size() = %zd, cpu_dists.size() = %zd\n", gpu_dists.size(), cpu_dists.size());
			for(size_t i = 0; i < cpu_dists.size(); ++i) {
				if(abs(gpu_dists[i] - cpu_dists[i]) > threshold) {
					equal = false;
					printf("%zd: gpu=%f, cpu=%f\n", i, gpu_dists[i], cpu_dists[i]);
				}
			}
			printf("Verification: %s\n", equal ? "valid" : "invalid");*/
		}
	}
	if(timings)
		fclose(timings);
	return 0;
}