#pragma once
#include"load.h"

/*
 * 主要用于施加作用在全局节点上的载荷，如重力载荷等
 */
typedef struct GlobalLoad:BasicLoad
{
	//ls:2020-04-06
	int LCIDDR;
	double XC;
	double YC;
	double ZC;
	int CID;
	//

	NodeManager* nodeManager;

	virtual void getLoadObject(SetManager* setMag, SurfaceManager* surfMag, vector<Part>& partMag, NodeManager* nodeMag) override;

	virtual void applyLoadGpu() override;

	virtual void applyLoadCpu() override;

	virtual ~GlobalLoad() { ; }
}GlobalLoad;