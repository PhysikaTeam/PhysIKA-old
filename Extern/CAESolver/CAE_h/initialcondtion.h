#pragma once
#include "node.h"
#include"set.h"
#include"part.h"
#include"rigid.h"
#include "rigidmanager.h"

struct RigidManager;
struct RigidCuda;
struct NodeCuda;
struct Part;

typedef struct InitialCondition
{
	enum Type
	{
		vel, disp
	};

	/**
	初始类型
	*/
	Type type;

	/**
	施加初始值的集合
	*/
	int nodeId;
	int nodeSetId;
	int partId;

	int NSIDEX;
	int BOXID;
	int IRIGID;
	int ICID;

	/**
	设定的初始值
	*/
	double value[6];

	/**
	single node initial
	*/
	NodeCuda *node;

	/**
	节点集合初始值
	*/
	Set *node_set;

	/**
	部件初始值
	*/
	Part *part;

	/**
	初始条件作用在刚体上
	*/
	RigidCuda *rigid;

	/**
	除刚体外被作用初始条件的所有节点
	*/
	vector<NodeCuda *> node_array_cpu;

	InitialCondition();

	/**
	生成初始条件节点集合
	*/
	void InitialNodeCreate();

	/**
	施加初始条件
	*/
	void setInitialCondition(RigidManager *rigidManager);
} InitialCondition;
