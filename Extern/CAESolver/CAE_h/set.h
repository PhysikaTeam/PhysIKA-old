#pragma once
#include<vector>
#include"cuda_runtime.h"
#include"helper_cuda.h"
#include"element.h"
#include<string>

using std::string;
using std::vector;

struct Part;
struct SegmentCuda;
struct NodeCuda;
struct NodeManager;
struct Element;
struct FEMDynamic;

typedef struct Set
{
	enum Type
	{
		nodeSet,
		segmentSet,
		partSet,
		elementSet
	};

	//ls:2020-03-18
	string title;
	int partsetId;
	int nodesetId;
	/**
	First,Second,Third,Fourth node attribute default value
	*/
	double A[4];
	double DA[4];
	int PID[8];
	int e[7];
	string solver;
	//specified entity
	vector<int>spentity_array;
	vector<int>nodesetId_array;  //ls:2020-04-06
	vector<int>partsetId_array;  //ls:2020-04-06
	vector<int>elemsetId_array;  //ls:2020-04-06
	//

	//ls:2020-04-06
	string option;
	//


	/**
	set的类型
	*/
	Type type;

	/**
	set的id
	*/
	int id;

	/**
	分布值
	*/
	double distribut_value[4];

	/**
	集合中成员的id
	*/
	vector<int> id_array;

	string name;

	/*
	cpu端成员集合
	*/
	vector<NodeCuda*> node_array_cpu;
	vector<SegmentCuda*> segment_array_cpu;
	vector<Part*> part_array_cpu;
	vector<Element*> element_array_cpu;

	/**
	gpu端节点成员的集合
	*/
	NodeCuda** node_array_gpu;
	SegmentCuda** segment_array_gpu;
	Element** element_array_gpu;

	Set();

	/*
	 * 集成cpu上的组件
	 */
	void assemblySetUnitCpu(FEMDynamic* domain);

	/*
	 * 集成gpu上的组件
	 */
	void assemblySetUnitGpu(FEMDynamic *domain);

	~Set();
} Set;

