#pragma once
#include"load.h"

/*
* ls:2020-04-06
*/
typedef struct SegmentLoad :BasicLoad
{
	//ls:2020-04-06
	//LOAD_SEGMENT_SET
	int id;
	string title;
	int SSID;
	int LCID;
	double SF;
	double AT;
	//int N1, N2, N3, N4, N5, N6, N7, N8;
	//

	NodeManager* nodeManager;

	//ls:focus

	virtual void getLoadObject(SetManager* setMag, SurfaceManager* surfMag, vector<Part>& partMag, NodeManager* nodeMag) override;

	virtual void applyLoadGpu() override;

	virtual void applyLoadCpu() override;

	virtual ~SegmentLoad();
	//
}SegmentLoad;