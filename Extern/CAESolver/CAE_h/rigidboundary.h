#pragma once
#include"boundary.h"
#include"rigid.h"

struct RigidCuda;

typedef struct RigidBoundary:BasicBoundary
{
	RigidCuda* rigid;

	//bool isPart;  ls:2020-04-24

	virtual void getBoundaryObject(SetManager* setMag, vector<Part>& partMag, NodeManager* nodeMag,
		RigidManager *rigidMag = nullptr, SurfaceManager* surMag = nullptr) override;

	virtual void imposebBounGpu(double dt, double previous_dt) override;

	virtual void imposebBounCpu(double dt, double previous_dt) override;

	virtual ~RigidBoundary() { ; }

}RigidBoundary;