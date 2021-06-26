#pragma once
#include"node.h"
#include"nodemanager.h"
#include"rigid.h"
#include"part.h"
#include"cuda_runtime.h"
#include<string>
#include"rigidmanager.h"
#include"curve.h"
#include"set.h"
#include"setmanager.h"
#include"rigidmanager.h"
#include"surfacemanager.h"
#include"boundarytype.h"

using std::string;

struct SetManager;
struct Curve;
struct NodeCuda;
struct Set;
struct Part;
struct RigidCuda;
struct RigidManager;
struct SurfaceManager;

typedef struct Boundary
{
	/**
	载荷类型
	*/
	enum Type { disp, vel, accel, spc };
	Type type;

	int id;
	int set_id;
	int node_id;
	int part_id;

	int dof[6];
	int curveId;
	double curve_scale_factor;

	/**
	边界条件施加在单个节点上
	*/
	NodeCuda *node;

	/**
	边界条件施加在set中的所有节点上
	*/
	Set *set;

	/**
	边界条件施加在part的所有节点上
	*/
	Part *part;

	/**
	边界条件施加在刚体上
	*/
	RigidCuda *rigid;

	/**
	除刚体外承受边界条件的所有节点
	*/
	vector<NodeCuda *> node_array_cpu;

	/**
	gpu端承受边界条件的所用节点
	*/
	NodeCuda **node_array_gpu;
	int node_num;

	/**
	gpu端受载节点创建
	*/
	void gpuNodeArrayCreate(NodeManager *node_manager);


	/**
	cpu端受载节点创建
	*/
	void cpuNodeArrayCreate(NodeManager *node_manager, vector<Part> *partArray_, SetManager *set_manager);

	/**
	节点施加边界条件
	*/
	void imposebBounGpu(double dt, double previous_dt, CurveManager *curve_manager);
	void imposebBounCpu(double dt, double previous_dt, CurveManager *curve_manager);

	~Boundary()
	{
		checkCudaErrors(cudaFree(node_array_gpu));
		node_array_gpu = nullptr;
		node_array_cpu.clear();
	}
} Boundary;