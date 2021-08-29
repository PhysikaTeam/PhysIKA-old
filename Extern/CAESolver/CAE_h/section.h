#pragma once
//#include "structure_declaration.h"
#include"element.h"

class CrossSection
{
	int fileId_;

	int storeId_;

	int gaussPointNum_;

	int typeId;

	bool isHourglassControl_;

	double hourglassParameter_;
public:

	virtual ~CrossSection(){}

	__host__ __device__ virtual void computeElementLumpedMass();

	__host__ __device__ virtual void computeCharacteristicLength() = 0;

	__host__ __device__ virtual void computeElementTimeStep() = 0;

	__host__ __device__ virtual void setGaussPointPtr() = 0;

	__host__ __device__ virtual void computeHourglassForce() = 0;

	__host__ __device__ virtual void computeInternalForce(Element *elm, Material *mat,const double dt) = 0;

	__host__ __device__ virtual void computeL2GMatrix(double** ndco, double** g2Lmat) = 0;

	__host__ __device__ virtual void computeBMatrix() = 0;
};