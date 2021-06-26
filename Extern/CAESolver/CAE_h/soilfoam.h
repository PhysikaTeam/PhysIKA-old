#pragma once
#include"materialnew.h"

typedef struct SoilFoamMaterial:MaterialNew
{
	double density_;
	double youngModulus_;
	double poissonRatio_;

	double shearModulus_;
	double bulkModulus_;

	double a1_;
	double a2_;
	double a3_;

	double initialYieldStress_;

	int compressCurveId_;
	double2 compressCurve_[CurveMaxNodeNum];

	SoilFoamMaterial();

	SoilFoamMaterial(Material *oldMat);

	__host__ __device__ virtual void createMaterialStatusInstance(GaussPoint* gp) override;
}SoilFoamMaterial;