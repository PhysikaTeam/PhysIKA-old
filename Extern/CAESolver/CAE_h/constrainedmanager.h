#pragma once
#include"constrained.h"

struct Constrained;

typedef struct ConstrainedManager
{
	vector<Constrained> constrained_array;
	Constrained *constrained_array_gpu;
	int constra_num;

	/**
	所有（焊接）从节点构成的集合
	*/
	NodeCuda** slave_node_array_gpu;
	int slave_node_num;

	/**
	所有球形关节（铰接）从节点（不分主从）构成的集合
	*/
	NodeCuda** joint_snode_array_gpu;
	int joint_snode_num;

	ConstrainedManager();

	/**
	焊接从cpu复制到gpu
	*/
	void gpuConstrainedCpy();

	/**
	焊接:力计算
	*/
	void allConstForCalGpu();
	void allConstForCalCpu();

	/**
	焊接:速度计算
	*/
	void allConstVelCalGpu();
	void allConstVelCalCpu();

	/**
	生成从节点集合
	*/
	void creatSlaveNodeArray();

	/**
	gpu执行所有(焊接)从节点的力计算
	*/
	void allConstForImproveCalGpu();

	/**
	gpu执行所有(焊接)从节点的速度计算
	*/
	void allConstVelImproveCalGpu();

	/**
	所有从节点的质量转移到主节点
	*/
	void allSlaveMassToMasterGpu();
	void allSlaveMassToMasterCpu();

	/**
	gpu执行所有球形关节（铰接）节点的力计算
	*/
	void allJointForImproveCalGpu();

	~ConstrainedManager();
}ConstrainedManager;