#pragma once
#include"materialnew.h"
#include"elasticmaterialstatus.h"

struct ElasticMaterialStatus;

typedef struct LinearElasticMaterial:MaterialNew
{
	double bulkModulus_;
	double shearModulus_;

	__host__ __device__ LinearElasticMaterial();

	__host__ __device__ LinearElasticMaterial(Material *oldMat);

	__host__ __device__ virtual void createMaterialStatusInstance(GaussPoint* gp)override;

	__host__ __device__ virtual void linkMatStatuToGaussPoint(GaussPoint* gp, MaterialStatus* matStaArray) override;

	__host__ __device__ virtual void computeStressMatrix3D(
		GaussPoint *gp, double stress[3][3], double deformGrad[3][3], double dt)const override;

	__host__ __device__ virtual void computeStressMatrix3D(
		GaussPoint* gp, double stress[3][3], double strainIncre[6], double spins[3], double dt) const override;

	__host__ __device__ virtual void computeStressMatrixPlaneStrain(
		GaussPoint* gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const override;

	__host__ __device__ virtual void computeStressMatrixPlaneStress(
		GaussPoint* gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const override;

	__host__ __device__ virtual void computeStressMatrix1d(
		GaussPoint* gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const override;

	__host__ __device__ virtual void reverseConstitutiveMatrix(GaussPoint* gp, double conMat[6][6])const override;

	__host__ virtual void createMatStatusArrayCpu() override;

	__host__ virtual void createMatStatusArrayGpu(const int gpu_id) override;

}LinearElasticMaterial;