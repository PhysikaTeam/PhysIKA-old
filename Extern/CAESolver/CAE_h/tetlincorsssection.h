#pragma once

#include"basesolidcrosssection.h"

class TetLinCrossSection:public BaseSolidCrossSection
{
public:
	__host__ __device__ virtual void computeInternalForce() override;
};