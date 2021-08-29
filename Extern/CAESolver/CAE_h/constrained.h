#pragma once
#include"node.h"
#include"set.h"
#include"setmanager.h"
#include"nodemanager.h"

typedef struct Constrained
{
	enum Type
	{
		spotweld, spline, extra_node,
		//ls:2020-03-18
		rigid, joint_spherical, joint_revolute, joint_cylindrical, joint_universal
		//
	};
	Type type;

	//ls:2020-03-18
	//joint
	int jointId;
	string HEADING;
	int node1;
	int node2;
	int node3;
	int node4;
	int node5;
	int node6;
	double RPS;
	double DAMP;

	int coordinateId;
	double timeFailure;
	double COUPL;
	double NXX;
	double NYY;
	double NZZ;
	double MXX;
	double MYY;
	double MZZ;

	bool hangFlag;
	
	//void mass_exam();
	//-----------------------
	//

	//ls:2020-04-06
	//CONSTRAINED_NODAL_RIGID_BODY
	int PID;
	int CID;
	int NSID;
	int PNODE;
	int IPRT;
	int DRFLAG;
	int RRFLAG;

	//spotweld---------------
	int N1;
	int N2;
	double EP;
	//

	/**
	节点集合Id
	*/
	int setId;

	/**
	约束id
	*/
	int id;

	/**
	spotweld
	*/
	double failureNormalForce;
	double failureShearForce;
	double exponentNormalForce;
	double exponentShearForce;
	double failureTime;

	/**
	包括主节点从节点在内的所有节点数目
	*/
	int node_num;

	/**
	形成焊点的集合
	*/
	Set *set;

	/**
	主节点
	*/
	NodeCuda *master_node_cpu;
	vector<NodeCuda *> joint_mnode_array_cpu;

	/**
	gpu主节点
	*/
	NodeCuda **master_node_gpu;

	/**
	从节点集合
	*/
	vector<NodeCuda *> slave_node_array_cpu;
	vector<NodeCuda *> joint_snode_array_cpu;

	/**
	gpu从节点
	*/
	NodeCuda** slave_node_array_gpu;

	/**
	cpu计算主节点质量
	*/
	void calConstraMassCpu();

	/**
	gpu计算主节点质量
	*/
	void calConstraMassGpu();

	/**
	修改从节点速度
	*/
	__host__ __device__ void modifySlaveSpeed(int slave_node_num);

	/**
	修改主节点力
	*/
	__host__ __device__ void modifyMasterForce(int slave_node_num);

	/**
	生成cpu端节点数据
	*/
	void cpuNodeCreate(SetManager *setManager);

	/**
	为从节点确定主节点编号
	*/
	void verMasterIdForSlave();

	/**
	为铰接从节点确定主节点编号
	*/
	void verJointMasterIdforSlave();

	/**
	生成gpu端数据
	*/
	void gpuNodeCreate(NodeManager *nodeManager);

	/**
	gpu端确定从节点中的主节点指针
	*/
	void gpuVerMasterPtr(NodeCuda* node_array_gpu);

	~Constrained();
}Constrained;