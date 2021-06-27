#pragma once
#include"load.h"

/*
 * 节点载荷默认为集中力载荷或者转矩载荷
 */
typedef struct NodeLoad:BasicLoad
{
	/*
	 * 载荷施加在节点集上
	 */
	Set* nodeSet_;

	/*
	 * 载荷施加在单个节点上
	 */
	NodeCuda* nodeCpu_;
	NodeCuda** nodeGpu_;

	virtual void getLoadObject(SetManager* setMag, SurfaceManager* surfMag, vector<Part>& partMag, NodeManager* nodeMag) override;

	virtual void applyLoadGpu() override;

	virtual void applyLoadCpu() override;

	virtual ~NodeLoad();
}NodeLoad;