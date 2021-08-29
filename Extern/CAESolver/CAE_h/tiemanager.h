#pragma once
#include"tie.h"
#include "surfacemanager.h"
#include"nodemanager.h"
#include<set>

using std::set;

struct NodeManager;
struct SurfaceManager;
struct Tie;

typedef struct TieManager
{
	int tie_num;

	vector<Tie> tie_array;
	
	set<int> slave_nodeId_array;
	
	set<int> master_nodeId_array;

	TieManager();

	/**
	计算绑定中力的传递
	*/
	void allTieAddSlaveNodeForceToMasterGpu();
	void allTieAddSlaveNodeForceToMasterCpu();

	/**
	计算绑定中速度的传递
	*/
	void allTieModifySlaveVelByMasterGpu();
	void allTieModifySlaveVelByMasterCpu();

	/**
	转移所有绑定对的质量
	*/
	void allTieTransferMassGpu();
	void allTieTransferMassCpu();

	/**
	增加一对绑定对
	*/
	void createNewTie(std::string masterSurface, std::string slaveSurface, const double position_tolerance,
		const int masterSurfaceId, const int slaveSurfaceId);

	/**
	将绑定的两个面串联起来
	*/
	void linkTieSurface(SurfaceManager *surfaceManager);

	/**
	判断是否存在一个面参与多个绑定
	*/
	void judgeSurfaceInNumTie();

	/**
	生成所有绑定从节点与对应的主块
	*/
	void produceAllSubTie();

	/**
	计算所有绑定面的插值函数
	*/
	void calAllTieInterpolateFun();

	/**
	生成Gpu上的所有绑定对
	*/
	void produceAllTieArrayGpu(NodeManager *nodeManger);

	/**
	检查所有绑定，确保没有重复绑定的情况
	*/
	void verfAllTie();
}TieManager;