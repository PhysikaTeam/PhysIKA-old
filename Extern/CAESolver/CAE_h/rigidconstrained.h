#pragma once
#include"rigid.h"
#include"boundary.h"
#include"curvemanager.h"

struct RigidCuda;
struct BasicBoundary;
struct CurveManager;

enum RigidConstraType
{
	rSpc,
	rDisp,
	rVel,
	rAccel,
};

typedef struct RigidConstraCuda
{
	RigidConstraType rcType_;

	int rigidId;

	int curveId_;

	int boundaryId_;

	int dof[6];

	double value;

	double scaleFactor[6];

	double spc[6];

	double directCosines[9];

	Curve* curve_;

	RigidCuda *rigid;

	BasicBoundary* parentBoundary;

	/**
	施加约束
	*/
	void rigidConstraint();

	/*
	 * 施加刚体上的强制边界条件
	 */
	void imposeBoundaryToRigid(const double dt, const double previous_dt);

	/*
	 * 连接到相应曲线
	 */
	void linkCurveToRigidConstra(CurveManager* cvMag);

	/**
	约束设置
	*/
	void initialRigidConstra(vector<RigidCuda> &rigid_array);

	/**
	刚体设置
	*/
	/*void rigidSet(vector<RigidCuda> &rigid_array);*///整合到到一个函数中

	/*
	 * 计算曲线值
	 */
	void computeRigidConstraValue(const double time);
} RigidConstraCuda;
