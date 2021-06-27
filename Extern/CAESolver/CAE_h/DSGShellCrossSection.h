#pragma once

#include "baseshellsection.h"

class DSGShellCrossSection:public BaseShellCrossSection
{
public:
	__host__ __device__ virtual void computeInternalForce() override;
};