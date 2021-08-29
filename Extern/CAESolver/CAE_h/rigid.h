#pragma once
#include "node.h"
#include "part.h"
#include "set.h"
#include "nodemanager.h"
#include <string>
#include <vector>
#include "rigidmanager.h"

using std::string;
struct RigidManager;

typedef struct RigidCuda
{
	/**
	cpu端节点
	*/
	vector<NodeCuda *> node_array_cpu;

	/**
	gpu端节点
	*/
	NodeCuda** node_array_gpu;

	/**
	cpu端节点
	*/
	vector<NodeCuda*> extra_node_array_cpu;
	vector<NodeCuda*> rigids_node_array_cpu;

	/**
	gpu端节点
	*/
	NodeCuda** extra_node_array_gpu;
	NodeCuda** rigids_node_array_gpu;

	/**
	刚体的母part
	*/
	Part *parentPart;

	/**
	刚体的母set
	*/
	Set *parentSet;

	/**
	节点数目
	*/
	int node_num;

	int refNodeId;

	int FinishRigidInitial;
	int ismergerigid;

	NodeCuda* refNode;

	int dofRigidBoundSign[6];
	double mass[6];
	double moment_of_inertia[3][3];
	double disp[6];
	double dispIncre[6];
	double accel[6];
	double position_of_rotCenter[3];
	double rotAxis[3];
	double rot_vel_center[3];
	double joint_force[6];
	double fext[6];
	double fint[6];

	double3 position;
	double3 tra_vel;
	double3 rot_vel;

	/**
	gpu端刚体数据
	*/
	double *rot_gpu;

	/**
	用于收集刚体所受的力
	*/
	double *force_gpu;
	double *force_cpu;

	/**
	cpu刚体节点列生成
	*/
	void rigidCpuNodeCreate(NodeManager *node_manager);

	/**
	gpu刚体节点列生成
	*/
	void rigidGpuNodeCreate(NodeManager *node_manager);

	/**
	刚体质量求解
	*/
	/*void rigidMass();*/
	void rigidMass(RigidManager *rigidManager);
	/**
	加速度更新
	*/
	void rigidAccelUpdate();

	/**
	刚体整体速度更新
	*/
	void rigidVelUpdate(const double dt, const double previous_dt);

	/**
	刚体内节点更新
	*/
	void rigidNodeVelUpdateGpu(double dt);
	void rigidNodeVelUpdateCpu(const double dt);

	/**
	刚体合力计算
	*/
	void rigidBodyForceCalGpu();
	void rigidBodyForceCalCpu();

	/**
	分配零拷贝内存
	*/
	void allZeroMemRigid();

	/**
	将刚体中所有节点标记为刚体
	*/
	void allNodeMark();

	/**
	清除掉刚体中节点的厚度
	*/
	void clearNodeThick();

	virtual ~RigidCuda();
} RigidCuda;
