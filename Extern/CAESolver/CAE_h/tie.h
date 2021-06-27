#pragma once
#include"node.h"
#include<set>
#include"nodemanager.h"

using std::set;

typedef struct SubTie
{
	int seg_store_id;
	int node_store_id;

	SegmentCuda *master_segment;
	NodeCuda *slave_node;
	double interp_funtion[4];

	/**
	链接gpu端的节点以及块
	*/
	__host__ __device__ void linkNodePtr(NodeCuda *node_array);
	__host__ __device__ void linkSegmentPtr(SegmentCuda *segment_array);

	/**
	求解插值函数
	*/
	__host__ __device__ void calInterpFun();

	/**
	修改主块的力
	*/
	__device__ void addForToMasterGpu();
	__host__ void addForToMasterCpu();

	/**
	修改从节点的速度
	*/
	__host__ __device__ void modifySlaveVel();

	/**
	修改主块的质量
	*/
	__device__ void transferMassGpu();
	__host__ void transferMassCpu();
}SubTie;

typedef struct Tie
{
	int masterSurface_id;
	int slaveSurface_id;

	std::string masterName;
	std::string slaveName;

	Surface *masterSurface;
	Surface *slaveSurface;

	enum Type
	{
		node_to_surface, surface_to_surface
	};
	/**
	类型默认为 node_to_surface
	*/
	Type type;

	/**
	形成绑定的距离
	*/
	double position_tolerance;

	/**
	adjust为 true 则将从节点移动到面上，默认为false
	*/
	bool adjust;

	/**
	为true 则不修改旋转自由度，默认为false
	*/
	bool is_no_rotation;

	/**
	为true 则无视壳的厚度，默认为false
	*/
	bool is_no_thickness;

	/**
	形成的子绑定对
	*/
	vector<SubTie> sub_tie_array_cpu;
	SubTie *sub_tie_array_gpu;

	/**
	生成绑定的从节点与对应的主面
	*/
	void produceSubTie(set<int> *slave_nodeid_array, set<int> *master_nodeid_array);

	/**
	生成gpu端的子绑定列
	*/
	void produceTieArrayGpu(NodeManager *node_manager);

	/**
	计算从节点再主块上的投影的插值函数
	*/
	void calInterpolateFunForSlave();

	/**
	将从节点的力添加到主块上
	*/
	void addSlaveForceToMasterGpu();
	void addSlaveForceToMasterCpu();

	/**
	依据主块的速度修改从节点的速度
	*/
	void modifySlaveVelByMasterGpu();
	void modifySlaveVelByMasterCpu();

	/**
	转移从节点的质量到主块上
	*/
	void transferAllSubTieMassGpu();
	void transferAllSubTieMassCpu();

	virtual ~Tie();
}Tie;
