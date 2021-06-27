#pragma once
#include"section.h"

class BaseShellCrossSection:public CrossSection
{
	bool isThickUpdate_;

	double initialThick_[4];

public:

	virtual ~BaseShellCrossSection(){}

 	__host__ __device__ virtual void setElementThick();

	__host__ __device__ virtual void computeCharacteristicLength() override;

	__host__ __device__ virtual void computeElementTimeStep() override;

	__host__ __device__ virtual void computeL2GMatrix(double** ndco, double** g2Lmat) override;

	__host__ __device__ virtual void computLocalCoord(double** ndco,double **g2LMat,double** lc);
};
