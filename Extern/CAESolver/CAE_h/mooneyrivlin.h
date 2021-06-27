#pragma once
#include"materialnew.h"

typedef struct MooneyRivlinMaterial:MaterialNew
{
	double constant1Hyelas_;
	double constant2Hyelas_;
	double penaltyCoeffLambda_;

	__host__ __device__ MooneyRivlinMaterial();

	__host__ __device__ MooneyRivlinMaterial(Material* oldMat);

	__host__ __device__ virtual void createMaterialStatusInstance(GaussPoint* gp) override;

	__host__ __device__ virtual void computeStressMatrix3D(GaussPoint* gp, double stress[3][3], double deformGrad[3][3], double dt)const override;

}MooneyRivlinMaterial;