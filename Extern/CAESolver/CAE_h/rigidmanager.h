#pragma once
#include"rigid.h"
#include"rigidconstrained.h"
//
#include"femdynamic.h"
#include"historynode.h"
//
#include<map>
#include<vector>

using std::map;
struct RigidCuda;
struct RigidConstraCuda;
struct Historynode;
//
struct FEMDynamic;
struct RigidExtraNodeManager;
//
typedef struct RigidManager
{
	//ls:2020-03-18
	/*
	刚体与节点（节点集）聚合关系
	*/
	int num_part;
	vector<int>partId_array;
	vector<int>nodeId_array;
	vector<int>nodeSetId_array;
	vector<bool>isSet_array;

	/**
	用于收集聚合节点集对刚体的反作用力
	*/
	double *rigid_force_add_gpu;
	double *rigid_force_add_cpu;

	/*
	cpu端数据
	*/
	vector<RigidCuda*> RigidPartArray_cpu;
	vector<NodeCuda*> NodeArray_cpu;
	/*
	gpu端数据
	*/
	RigidCuda** RigidPartArray_gpu;
	NodeCuda** NodeArray_gpu;
	//

	/**
	刚体集合
	*/
	vector<RigidCuda> rigid_array;

	/**
	刚体约束
	*/
	vector<RigidConstraCuda> rigidConstra_array;

	/**
	刚体聚合关系
	*/
	vector<int> masterRigidId;
	vector<int> slaveRigidId;
	vector<RigidCuda*> masterRigidArray;
	vector<RigidCuda*> slaveRigidArray;

	map<int, int> rigidMapToRefNode;

	__host__ __device__ RigidManager();

	/**
	存储主、从刚体(merge rigid)
	*/
	void masterRigidToSlaveRigid_pre(RigidExtraNodeManager *rigid_extra_node_manager);

	/**
	根据从刚体修改主刚体质量、质心和惯性量
	*/
	void masterRigidToSlaveRigid_mass();

	/**
	根据主刚体修改从刚体的运动
	*/
	void masterRigidToSlaveRigid_vel();

	/**
	根据从刚体修改主刚体的力
	*/
	void masterRigidToSlaveRigid_force();

	//ls:2020-06-24 added
	/**
	依据主刚体运动数据修改关联节点运动数据(准备)
	*/
	void RigidPartToSlaveNodes_pre(FEMDynamic *domain, RigidExtraNodeManager *rigid_extra_node_manager);
	
	/**
	依据主刚体运动数据修改关联节点运动数据(已被修改为：按刚体节点速度更新方式更新)
	*/
	void RigidPartToSlaveNodesGpu(FEMDynamic *domain);
	
	/*	
	数据copy
	*/
	void RigidManager::RigidToSlaveNodesCraetGpu(FEMDynamic *domain);
	//
	
	/**
	依据从节点力更新刚体运动
	*/
	void SlaveNodesForceToRigidGpu(RigidCuda& rigid_array);

	/*
	 * 将刚体所有节点置标志位
	 */
	void setRigidFlagForAllNode();

	/*
	 * 将刚体中所有节点的厚度清除，以避免影响接触
	 */
	void clearRigidAllNodeThick();

	//ls:2020-06-24 added
	~RigidManager();
	//

} RigidManager;
