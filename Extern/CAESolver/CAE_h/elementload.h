#pragma once
#include"load.h"

/*
 * 单元载荷目前默认为体载荷
 */
typedef struct ElementLoad:BasicLoad
{
	//ls:2020-06-01
	int id;
	string title;
	int EID;
	int LCID;
	double SF;
	double AT = 0;
	//

	/*
	 * 载荷施加在单元集合上
	 */
	Set* elementSet_;

	/*
	 * 单元施加在单个单元上
	 */
	Element* elementCpu_;
	Element* elementGpu_;

	virtual void getLoadObject(SetManager* setMag, SurfaceManager* surfMag, vector<Part>& partMag, NodeManager* nodeMag) override;

	virtual void applyLoadGpu() override;

	virtual void applyLoadCpu() override;

	virtual ~ElementLoad();
}ElementLoad;