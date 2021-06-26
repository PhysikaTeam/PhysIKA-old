//#pragma once
//#include "beam2nd.h"
//#include"shell2nd.h"
//#include"solid2nd.h"
//#include"materialold.h"
//#include"nodemanager.h"
//#include"sectionold.h"
//
//struct Section;
//struct NodeManager;
//struct BeamElement2nd;
//struct ShellElement2nd;
//struct SolidElement2nd;
//struct Material;
//
//typedef struct Elm2ndManager
//{
//	vector<BeamElement2nd> beam2ndArrayCpu_;
//	BeamElement2nd* beam2ndArrayGpu_;
//
//	vector<ShellElement2nd> shell2ndArrayCpu_;
//	ShellElement2nd* shell2ndArrayGpu_;
//
//	vector<SolidElement2nd> solid2ndArrayCpu_;
//	SolidElement2nd* solid2ndArrayGpu_;
//
//	int nodePerElement;
//	int node_amount;
//	int elementAmount;
//
//	/**
//	单元长度最小值最大值
//	*/
//	double *eign_cpu;
//	double *eign_gpu;
//
//	/**
//	单元管理器中的总质量
//	*/
//	double tot_mass;
//
//	/**
//	单元质量矩阵计算
//	*/
//	void calElementMass(Material *material, Section *section);
//
//	/**
//	cpu单元节点数据匹配
//	*/
//	void elementNodeMatch(NodeManager *node_manager);
//
//	/**
//	gpu端单元数据创建
//	*/
//	void gpuElementCreate(const Section *section);
//
//	/**
//	gpu节点数据匹配
//	*/
//	void gpuNodeMatch(NodeManager *node_manager, const Section *section);
//
//	/**
//	gpu端数据清除
//	*/
//	void gpuDataClear();
//
//	/**
//	节点厚度分配
//	*/
//	void shellNodeThickAndRadius(Section *section);
//
//	/**
//	求解所有单元内力
//	*/
//	void allElementInterForceCalGpu(Material *material, Section *section, double dt);
//	void allElementInterForceCalCpu(Material *material, Section *section, double dt);
//
//	/**
//	根据预索引，建立所有单元与节点之间的索引信息
//	*/
//	void bulidElementIndexNodeGpu();
//
//	/**
//	确定管理器中所有单元的材料属性编号
//	*/
//	void verElementMatSecId(const int mater_id, const int section_id);
//
//	/**
//	分配零拷贝内存
//	*/
//	void allocZeroMemEle();
//}Elm2ndManager;