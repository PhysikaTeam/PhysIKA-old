#pragma once
#include"cuda_runtime.h"
#include"materialold.h"
#include"gausspoint.h"
#include"materialstatus.h"
#define MaxGpuNum 8
#define CurveMaxNodeNum 128

struct MaterialStatus;

typedef struct MaterialNew
{
	int id;

	MaterialNewType mType;

	int dynaTypeId;

	double solidWaveNumber_;

	double shellWaveNumBer_;

	double beamWaveNumBer_;

	double discreteWaveNumber_;

	double density_;

	double youngModulus_;

	double poissonRatio_;

	int matStatusNum;

	MaterialStatus* matStatusArrayCpu_;
	MaterialStatus* matStatusArrayGpu_[MaxGpuNum];

	__host__ __device__ MaterialNew();

	__host__ __device__ MaterialNew(int i, MaterialNewType t = MaterialNewType::mNull, int di = 0);

	__host__ __device__ MaterialNew(Material* oldMat);

	__host__ __device__ void rotateStress(double stress[6],double spins[3]);

	__host__ __device__ void computeStressCoordSysConverse(double stress[6],double* localCoordSys[3]);

	__host__ __device__
	virtual void computeStressMatrix3D(GaussPoint *gp, double stress[3][3], double strainIncre[6], double spins[3], double dt)const
	{
		printf("here not executed\n");
	}

	__host__ __device__
	virtual void computeStressMatrix3D(GaussPoint *gp, double stress[3][3], double deformGrad[3][3], double dt)const
	{
		printf("here not executed\n");
	}

	__host__ __device__
	virtual void computeStressMatrixPlaneStrain(GaussPoint *gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const
	{
		printf("here not executed\n");
	}

	__host__ __device__
	virtual void computeStressMatrixPlaneStress(GaussPoint *gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const
	{
		printf("here not executed\n");
	}

	__host__ __device__
	virtual void computeStressMatrix1d(GaussPoint *gp, double stress[], double strainIncre[], double* localCoordSys[3], double dt)const
	{
		printf("here not executed\n");
	}

	__host__ __device__ virtual void computeWaveNumber();

	__host__ __device__ virtual void reverseConstitutiveMatrix(GaussPoint *gp, double conMat[6][6])const
	{
		printf("here not executed\n");
	}

	__host__ __device__ 
	virtual void createMaterialStatusInstance(GaussPoint *gp)
	{
		printf("here not executed\n");
	}

	__host__ __device__
	virtual void linkMatStatuToGaussPoint(GaussPoint *gp,MaterialStatus *matStaArray)
	{
		printf("here not executed\n");
	}

	__host__ virtual void createMatStatusArrayCpu()
	{
		printf("here not executed\n");
	}

	__host__ virtual void createMatStatusArrayGpu(const int gpu_id=0)
	{
		printf("here not executed\n");
	}

	virtual ~MaterialNew();
}MaterialNew;