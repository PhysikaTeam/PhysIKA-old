#pragma once
#include"basebeamcrosssection.h"

class BSBeamCrossSection:public BaseBeamCrossSection
{
public:
	__host__ __device__ virtual void computeInternalForce() override;
};