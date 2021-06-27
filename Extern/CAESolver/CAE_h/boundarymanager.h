#pragma once
#include"boundary.h"
#include"basicboundary.h"

struct FEMDynamic;

typedef struct BoundaryManager
{
	/*vector<Boundary> nodeBoun_array;*/

	vector<BasicBoundary*> boundaryArray_;

	void pushBack(BasicBoundary* bdPtr);

	void linkCurveToAllBoundary(CurveManager* cvMag);

	void computeAllBoundaryValue(double time);

	void getAllObjectForBoundary(FEMDynamic* domain);

	void imposeAllBoundaryCpu(double dt,double previos_dt);

	void imposeAllBoundaryGpu(double current_time, double dt, double previos_dt); //ls:2020-06-14 modified 

	~BoundaryManager()
	{
		for(auto& ib:boundaryArray_)
		{
			delete[] ib;
		}
	}

} BoundaryManager;