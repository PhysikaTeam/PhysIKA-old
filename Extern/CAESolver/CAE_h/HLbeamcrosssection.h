#pragma once
#include"basebeamcrosssection.h"

class HLBeamCrossSection:public BaseBeamCrossSection
{
public:
	__host__ __device__ virtual void computeInternalForce() override;
};