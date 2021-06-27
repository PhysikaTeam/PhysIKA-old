#pragma once
#include"cuda_runtime.h"
struct ContactNodeCuda;
struct SegmentCuda;

typedef struct ContactPairCuda
{
	enum Type
	{
		null, node_to_segment, node_to_edge, node_to_node
	};

	/**
	组成接触对的接触点与接触块
	*/
	/*int contact_node_id;*////@是否彻底删除还需进一步观察
	int segment_global_id;
	ContactNodeCuda* contactNode;
	SegmentCuda* contact_segment;

	/**
	形成接触对的边或者点
	*/
	int edgeOrNode;
	int position;
	int count;

	/**
	点到接触元素的距离
	*/
	double dist;

	/**
	接触对的类型
	*/
	Type type;

	double normVec[3];
	
	double fcn[2];

	double frictionForce[2];
	/**
	采用防御节点法时的防御节点的质量与插值形函数
	*/
	double subMass[4];
	double shape[4];

	/*
	 * 采用罚函数法时表面刚度
	 */
	double interfaceStiff;

	/**
	双向搜素接触对
	*/
	__device__ void searchContactPairBilateralGpu(int *contact_pair_num_gpu);

	/**
	防御节点法穿透量计算
	*/
	__device__ void penetraCalDefenGpu(int search_type, int *contact_pair_num);
	__device__ void nodeToSegmentPenetraCalDefenGpu(int search_type, int *contact_pair_num);
	__device__ void nodeToEdgePenetraCalDefenGpu(int search_type, int *contact_pair_num);
	__device__ void nodeToNodePenetraCalDefenGpu(int search_type, int *contact_pair_num);

	/**
	接触力计算
	*/
	/**
	未解耦的防御节点法
	*/
	__device__ void defenceNodeCalGpu(const double dt, const int mstp, int *contact_pair_num, int search_type);
	__device__ void nodeToSegmentForceGpu(const double dt, const int mstp, int *contact_pair_num, int search_type);
	__device__ void nodeToEdgeForceGpu(const double dt, const int mstp, int *contact_pair_num, int search_type);
	__device__ void nodeToNodeForceGpu(const double dt, const int mstp, int *contact_pair_num, int search_type);

	__host__ void defenceNodeCalCpu(const double dt, const int mstp, int *contact_pair_num);
	__host__ void nodeToSegmentForceCpu(const double dt, const int mstp, int *contact_pair_num);
	__host__ void nodeToEdgeForceCpu(const double dt, const int mstp, int *contact_pair_num);
	__host__ void nodeToNodeForceCpu(const double dt, const int mstp, int *contact_pair_num);

	/**
	解耦的防御节点法
	*/
	__device__ void defenceNodeCalGpuDecouple(const double dt, const int mstp, int *contact_pair_num, int search_type);
	__device__ void nodeToSegmentForceGpuDecouple(const double dt, const int mstp, int *contact_pair_num, int search_type);
	__device__ void nodeToEdgeForceGpuDecouple(const double dt, const int mstp, int *contact_pair_num, int search_type);
	__device__ void nodeToNodeForceGpuDecouple(const double dt, const int mstp, int *contact_pair_num, int search_type);

	///**
	//双向防御节点法
	//*/
	///**
	//穿透量计算
	//*/
	//__device__ void penetraCalBilateralGpu();
	//__device__ void nodeToSegmentpenetraCalBilateralGpu();
	//__device__ void nodeToEdgepenetraCalBilateralGpu();
	//__device__ void nodeToNodepenetraCalBilateralGpu();

	//__device__ void defenceNodeCalBilateralGpu(const double dt, const int mstp, int *contact_pair_num);
	//__device__ void nodeToSegmentForceBilateralGpu(const double dt, const int mstp, int *contact_pair_num);
	//__device__ void nodeToEdgeForceBilateralGpu(const double dt, const int mstp, int *contact_pair_num);
	//__device__ void nodeToNodeForceBilateralGpu(const double dt, const int mstp, int *contact_pair_num);

	/**
	罚函数法
	*/
	__device__ void penaltyContForCalGpu(const double dt, const int mstp, const int penalty_method,
		int *contact_pair_num, int search_type);
	__device__ void nodeToSegmentPenaContForGpu(const double dt, const int mstp, int *contact_pair_num,
		int penalty_method, int search_type);
	__device__ void nodeToEdgePenaContForGpu(const double dt, const int mstp, int *contact_pair_num,
		int penalty_method, int search_type);
	__device__ void nodeToNodePenaContForGpu(const double dt, const int mstp, int *contact_pair_num,
		int penalty_method, int search_type);

	/**
	接触后搜索
	*/
	__host__  __device__ void postSearch();
	__host__  __device__ void nodeToSegmentCudaPostSearch();
	__host__  __device__ void nodeToEdgePostSearch();
	__host__  __device__ void nodeToNodePostSearch();

	/*
	 * 根据罚函数方法确定接触对的接触刚度
	 */
	__host__ __device__ void computeInterForceStiff(double dt,const int penalty_method=0);

	/**
	记录fcn
	*/
	__host__ __device__ void storeFcn();

	/*
	* 将接触力保存到节点的dfn数组中
	*/
	__device__ void storeFcnToNodeDfn();
} ContactPairCuda;
