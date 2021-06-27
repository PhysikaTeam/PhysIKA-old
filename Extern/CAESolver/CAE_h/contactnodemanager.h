#pragma once
#include"surface.h"
#include <map>
#include"contactnode.h"
#include"contactmanager.h"
#include"helper_cuda.h"
#include"nodemanager.h"

struct NodeManager;
struct Surface;
struct Contact;
struct ContactNodeCuda;

typedef struct ContactNodeManager
{
	vector<ContactNodeCuda> contact_node_array_cpu;
	std::map<int, int> nodeLinkContactNodeCuda;

	/**
	接触点总数
	*/
	int contact_node_num;

	/**
	gpu接触点
	*/
	ContactNodeCuda *contact_node_array_gpu;

	/*
	* 多gpu接触点
	*/
	vector<ContactNodeCuda*> contactNodeArrayMultiGpu;

	ContactNodeManager();

	/**
	cpu接触节点生成
	*/
	void cpuContactNodeCudaCreate(vector<Surface> &surface_array,vector<Contact> &contactArray,NodeManager *ndMag);

	/**
	gpu接触点生成
	*/
	void gpuContactNodeCudaCreate(NodeManager *node_manager);
	void multiGpuContactNodeCreate(vector<int> gpuIdArray, NodeManager *ndMag = nullptr);
	void multiGpuNodeLinkContactNode(NodeManager* ndMag,vector<int>& gpuIdArray);

	/*
	* managed接触点生成
	*/
	void managedContactNodeCreate(NodeManager* node_manager);

	/*
	* 生成接触点的gpu编号，目前只能以随机数方式的来达到可能的负载均衡
	*/
	void calGpuIdForContactNodeMultiGpu(vector<int> gpuIdArray);

	/**
	整合接触力
	*/
	void integrateContactForceMultiGpu(const int cnPairNum, int gpu_id, cudaStream_t stream);
	void integrateContactForceGpu(const int contact_pair_num);
	void integrateContactForceCpu();

	/**
	计算所有碰撞点的罚因子
	*/
	//void calAllNodePenaFactorCpu(double penaScale, double dt);
	//void CalAllNodePenaFactorGpu(double penaScale, const double dt);

	/**
	清除所有标记
	*/
	void clearAllFlag();

	~ContactNodeManager();

} ContactNodeManager;
