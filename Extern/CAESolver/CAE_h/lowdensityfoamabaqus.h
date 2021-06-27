#pragma once
#include"materialnew.h"

typedef struct LowDensityFoamAbaqusMaterial:MaterialNew
{
	double mu1_;
	double mu2_;
	double alpha_;

	int tensionCruveId_;
	int compressCurveId_;

	double2 tensionCurve_[CurveMaxNodeNum];
	double2 compressCurve_[CurveMaxNodeNum];

	__host__ __device__ LowDensityFoamAbaqusMaterial();

	__host__ __device__ LowDensityFoamAbaqusMaterial(Material *oldMat);
};