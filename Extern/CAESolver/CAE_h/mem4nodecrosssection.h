#pragma once
#include"baseshellsection.h"

class Membrance4NodeCrossSection:public BaseShellCrossSection
{
	__host__ __device__ virtual void computeHourglassForce() override;

	__host__ __device__ virtual void computeInternalForce() override;
};