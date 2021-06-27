#pragma once
#include"load.h"

struct FEMDynamic;
struct BasicLoad;

typedef struct LoadManager
{
	vector<BasicLoad*> loadArray_;

	void pushBack(BasicLoad* newLoad);

	void getAllObjectForLoad(FEMDynamic* domain);

	void linkCurveToAllLoad(CurveManager* cvMag);

	void applyAllLoadCpu();

	void applyAllLoadGpu();

	void computeAllLoadValue(double time);

	~LoadManager();
} LoadManager;