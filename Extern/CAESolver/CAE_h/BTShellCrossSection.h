#pragma once
#include"baseshellsection.h"

class BTShellCrossSection:public BaseShellCrossSection
{
public:
	__host__ __device__ virtual void computeHourglassForce() override;

	__host__ __device__ virtual void computeInternalForce(Element *elm, Material *mat, const double dt) override;

	__host__ __device__ virtual void computeL2GMatrix(double** ndCo, double** g2Lmat) override;

	__host__ __device__ virtual void computLocalCoord(double** ndco, double** g2LMat, double** lc) override;
};

