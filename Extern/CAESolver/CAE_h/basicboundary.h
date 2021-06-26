#pragma once
#include"node.h"
#include"nodemanager.h"
#include"rigid.h"
#include"part.h"
#include"cuda_runtime.h"
#include<string>
#include"rigidmanager.h"
#include"curve.h"
#include"set.h"
#include"setmanager.h"
#include"rigidmanager.h"
#include"surfacemanager.h"
#include"boundarytype.h"

using std::string;
struct RigidManager;

typedef struct BasicBoundary
{
	BoundaryType bType;

	int file_id;

	int objectId_;

	int curveId_;

	int dof[6];

	bool isSet;

	//ls:2020-04-24 added
	bool isPart;
	//

	//ls:2020-06-14 added
	/*
	define birth and death time
	*/
	double birth_time;
	double death_time;
	//

	double value_;

	double scaleFactor[6];

	string curveName_;

	string objectName_;

	Curve* curve_;

	BasicBoundary();

	virtual void getBoundaryObject(SetManager* setMag, vector<Part>& partMag, NodeManager* nodeMag,
		RigidManager *rigidMag = nullptr, SurfaceManager* surMag = nullptr) = 0;

	void computeBoundaryValueAt(double time);

	void linkBoundaryCurve(CurveManager* cvMag);

	virtual void imposebBounGpu(double dt, double previous_dt) = 0;

	virtual void imposebBounCpu(double dt, double previous_dt) = 0;

	virtual ~BasicBoundary() { ; }

}BasicBoundary;
