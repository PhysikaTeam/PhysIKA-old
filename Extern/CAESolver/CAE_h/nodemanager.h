#pragma once
#include"cuda_runtime.h"
#include"node.h"
#include<map>
#include "helper_cuda.h"

using std::map;
struct MultiGpuManager;

typedef struct NodeManager
{
	/**
	cpu端数据
	*/
	vector<NodeCuda> node_array_cpu;

	/**
	gpu端数据
	*/
	NodeCuda* node_array_gpu;

	int dimesionPerNode;
	int dofPerNode;
	int totNodeNum;

	/**
	系统总动能
	*/
	double kinetic_energy;

	map<int, NodeCuda*> node_id_link;

	/*
	 * 清除所有节点应力应变
	 */
	void clearAllNodeStressStrainGpu();

	/*
	 * 平均所有节点的应力应变
	 */
	void averageNodeStressStrainGpu();

	/**
	信息初始化
	*/
	void factorInitial();

	/**
	输出给定id的节点
	*/
	NodeCuda* returnNode(int node_id);

	/**
	形成节点与id之间的映射
	*/
	void buildMap();

	/**
	gpu端数据创建
	*/
	void gpuNodeCreate();

	void multiGpuNodeCreate(MultiGpuManager *mulGpuMag);
	
	void managedNodeCreate();

	/**
	计算节点的等效质量
	*/
	void effectMassCal();

	/*
	* 质量检测
	*/
	void nodeMassCheckCpu();
	void nodeMassCheckGpu();

	/**
	重置节点质量
	*/
	void resetNodeMassGpu();
	void resetNodeMassCpu();

	/**
	所有节点加速度速度更新
	*/
	void allNodeAccelVelUpdateGpu(const double dt, double previous_dt);
	void allNodeAccelVelUpdateCpu(double dt, double previous_dt);

	/**
	所有节点位移更新
	*/
	void allNodeDispUpdateGpu(double dt);
	void allNodeDispUpdateCpu(double dt);

	/*
	* 根据节点编号排序节点
	*/
	void sortNodeOnFileId();

	/**
	gpu节点数据复制到cpu
	*/
	void gpuDataToCpu();

	/**
	根据预索引策略集合节点内力
	*/
	void addNodeInterForcePreIndex();

	/**
	保存节点的初始坐标
	*/
	void storeAllNodeIniCoord();

	/**
	保存节点的初始质量
	*/
	void storeNodeIniMassCpu();
	void storeNodeIniMassGpu();

	/**
	依据冯慧师姐的程序，增添的提前整合接触力的程序
	*/
	void improveContactEffectToDisp(double dt, double previous_dt);

	/**
	计算系统的动能
	*/
	void calKineticEnenrgy();

	/**
	质量缩放后更新节点的等效质量
	*/
	void updateEquivalentMassAfterScaleGpu();
	void updateEquivalentMassAfterScaleCpu();

	~NodeManager();

	NodeManager();
} NodeManager;