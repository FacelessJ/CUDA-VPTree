#pragma once

#include <vector>
#define DIM 2

struct CUDA_VPNode
{
	double threshold;
	int index;
	int left, right;

	CUDA_VPNode() : threshold(0.), index(0), left(-1), right(-1) {}
};

struct Point
{
	double coords[DIM];
};

double distance(const Point& a, const Point& b);


class CUDA_VPTree
{
public:
	CUDA_VPTree();
	~CUDA_VPTree();
	void createVPTree(std::vector<Point> &data);
	void search(const std::vector<Point> &queries, std::vector<int> &indices, std::vector<double> &distances);
	int treeDepth();
private:
	void buildFromPoints(std::vector<Point> &data);
	
	CUDA_VPNode *gpu_nodes;
	//int *gpu_indices; //What does this do?
	Point *gpu_points;
	
	size_t num_points;
	bool tree_valid;

	struct DistComparator
	{
		const Point& item;
		DistComparator(const Point& item) : item(item) {}
		bool operator()(const Point& a, const Point& b) {
			return distance(item, a) < distance(item, b);
		}
	};
};