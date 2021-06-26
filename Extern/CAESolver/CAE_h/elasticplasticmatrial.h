#pragma once
#include"materialnew.h"
#include"plasticmaterialstatus.h"

struct PlasticMaterialStatus;

enum ElasticPlasticMode
{
	
};

typedef struct ElasticPlasticMaterial:MaterialNew
{
	double bulkModulus_;
	double shearModulus_;
	double hardeningParameter_;
	double initialYieldStress_;
	double tangentModulus_;

	//指数塑性模型材料参数
	double strengthCoeffi_;
	double hardenExponent_;
	double elasticStrainToYield_;
	double hardeningModulus;

	int plasticModel_;
	int curveId_;
	int curvePointNum_;

	double yCurveStress[CurveMaxNodeNum];
	double xCurveStrain[CurveMaxNodeNum];

	__host__ __device__ ElasticPlasticMaterial();

	__host__ __device__ ElasticPlasticMaterial(Material *oldMat);

	__host__ __device__ 
	virtual void createMaterialStatusInstance(GaussPoint* gp) override;

	__host__ __device__ virtual void linkMatStatuToGaussPoint(GaussPoint* gp, MaterialStatus* matStaArray) override;

	__host__ __device__ 
	virtual void reverseConstitutiveMatrix(GaussPoint* gp, double conMat[6][6]) const override;

	__host__ __device__ 
	virtual void computeStressMatrix3D(GaussPoint* gp, double stress[3][3], double deformGrad[3][3], double dt)const override;

	__host__ __device__ 
	virtual void computeStressMatrix3D(GaussPoint* gp, double stress[3][3], double strainIncre[6], double spins[3], double dt) const override;

	__host__ __device__ 
	virtual void computeStressMatrixPlaneStrain(GaussPoint* gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const override;

	__host__ __device__ 
	virtual void computeStressMatrixPlaneStress(GaussPoint* gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const override;


	__host__ virtual void createMatStatusArrayCpu() override;
	__host__ virtual void createMatStatusArrayGpu(const int gpu_id) override;

	//__host__ __device__ 
	//virtual void computeStressMatrix1d(GaussPoint* gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const override;

private:

	__host__ __device__ void radialReturnAlgorith(double stress[3][3],double &yieldStress,double &effStrain,double stsTrial[3][3])const;

	__host__ __device__ void radialEvl_k_prime(const double pstn, double &ys_k_prime)const;

	__host__ __device__ void radialEvl_k(const double pstn, double &ys_k)const;

	__host__ __device__ void interpolatEquivalentStress(double ep,double &ak,double &qh)const;

}ElasticPlasticMaterial;