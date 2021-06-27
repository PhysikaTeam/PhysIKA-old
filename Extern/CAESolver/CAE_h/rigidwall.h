#pragma once
#include"cuda_runtime.h"
#include"node.h"
#include"femdynamic.h"
#include"curvemanager.h"

typedef struct RigidWall
{
	enum Gtype
	{
		flat, unlimited_flat, sphere, cylinder
	};

	enum Mtype
	{
		vel, disp
	};

	//ls:2020-03-18
	int id;
	string title;
	int NSID; //Nodal set ID containing slave nodes 
	int NSIDEX; //Nodal set ID containing nodes that are exempted as slave nodes
	int BOXID; //All nodes in box are included as slave nodes to rigid wall
	/*All nodes within a normal offset distance, OFFSET,
	to the rigid wall are included as slave nodes for the rigid wall*/
	double OFFSET; 
	double birth_time;
	double death_time;
	double RWKSF; // Stiffness scaling factor

	double XT;
	double YT;
	double ZT;
	double XH;
	double YH;
	double ZH;
	double FRIC;  //
	double WVEL;
	//Additional card for FORCES keyword option.
	int SOFT;
	int SSID;
	int N[4];
	// Additional card for FINITE keyword option
	double XHEV;
	double YHEV;
	double ZHEV;
	double LENL;
	double LENM;
	//
	int OPT;
	//

	/**
	刚性墙形状
	*/
	Gtype gtype;

	/**
	刚性墙类型
	*/
	Mtype mtype;

	/**
	被包含在刚性墙中的节点(ls:修改为slaveNode)
	*/
	int node_set_id;
	Set *node_set;

	/**
	法向向量的定义
	*/
	double3 nor_vec;
	double3 tan_vec_1;
	double3 tan_vec_2;

	/**
	向量起点
	*/
	double3 xt;
	/**
	向量终点
	*/
	double3 xh;

	/**
	定义有限平面所需的参数
	*/
	double3 xhev;
	double len_l;
	double len_m;

	/**
	定义圆面所需参数
	*/
	double rad_sph;

	/**
	定义圆柱面所需的参数
	*/
	double rad_cyl;
	double len_cyl;

	/**
	定义运动所需的参数
	*/
	int curve_id;
	Curve *curve;
	double3 vector_motion;
	double disp_value;
	double velocity;

	/**
	摩擦因子
	*/
	double friction_factor;

	//ls:2020-07-01 added
	/**
	从节点集合
	*/
	//vector<NodeCuda *> slave_node_array_cpu;
	/*
	刚性墙总质量
	*/
	double totMass;

	int TotslaveNodes;

	/**
	gpu从节点
	*/
	NodeCuda** slave_node_array_gpu;
	
	/*
	收集从节点
	*/
	void slaveNodeCollect(SetManager *setManager, FEMDynamic *domain);

	/*
	数据copy
	*/
	void gpuSlaveNodeCopy(FEMDynamic *domain);

	/*
	利用罚参数法/防御节点法计算接触力
	*/
	void calContactForce(FEMDynamic *domain, double current_time, double dt);

	/*
	刚性墙移动
	*/
	void movingRigidwall(CurveManager *curveManager);

	RigidWall();
	//

	/**
	求解法向向量与其他方向向量
	*/
	void calAllVector();

	/**
	控制刚性墙的运动
	*/
	__host__ __device__ void calWallMotion(const double dt, const double current_time);

	/**
	罚函数法计算接触力
	*/
	__host__ __device__ void calContactForcePenlty(ContactNodeCuda *contact_node);
} RigidWall;