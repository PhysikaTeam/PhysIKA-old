#pragma once
#include"boundary.h"
#include"rigidmanager.h"
#include"surfacemanager.h"
#include"part.h"
#include"nodemanager.h"
#include<string>

using namespace std;

struct NodeManager;
struct Part;
struct SurfaceManager;
struct RigidManager;
struct BasicBoundary;

typedef struct NodeBoundary:BasicBoundary
{
	//ls:2020-06-24 
	int id;
	string title;

	/*
	 * 在节点集合上施加强制边界条件
	 */
	Set* nodeSet_;

	/*
	 * 在单个节点上施加边界条件
	 */
	NodeCuda* nodeCpu_;
	NodeCuda** nodeGpu_;

	NodeBoundary();

	virtual void getBoundaryObject(SetManager* setMag, vector<Part>& partMag, NodeManager* nodeMag,
		RigidManager *rigidMag = nullptr, SurfaceManager* surMag = nullptr) override;

	virtual void imposebBounGpu(double dt, double previous_dt) override;

	virtual void imposebBounCpu(double dt, double previous_dt) override;

	virtual ~NodeBoundary();

}NodeBoundary;