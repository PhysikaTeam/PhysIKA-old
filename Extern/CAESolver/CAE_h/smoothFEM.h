#pragma once
#include"element.h"
#include"node.h"

typedef struct SmoothFEM :Element
{
	SmoothFEMType smoothFemType_;

	int* nodeIdArray_;
	NodeCuda** nodeArray_;

	int gpu_id;

	int material_id;
	int section_id;

	double dtEle_;
	double lengthChart_;

public:

	__host__ __device__ void setParameter(int matId,int secId,SmoothFEMType sType,int gpuId=0);

	__host__ __device__ void copyShapeFunctionPartial(double* shpkdx,double* shpkdy,double* shpkdz=nullptr);

	__host__ __device__ void bulidNodeLink(NodeCuda *nodeSysArray);

	__host__ __device__ void allocateNodeMem();

	__host__ __device__ void freeMemory();

	__device__ void computeSmoothFemInterForceGpu(MaterialNew *material, Section *section, HourglassControl *hgControl, const double dt);

	__device__ void calInterForSmoothFemTL(GaussPoint* gp, double* bMat[3], const MaterialNew* mat, double dt);

	__host__ __device__ void computeSmoothFemCharactLenght();

	__device__ void computeSmoothFemNodeStressStrainGpu();

	__host__ void computeSmoothFemNodeStressStrainCpu();

}SmoothFEM;