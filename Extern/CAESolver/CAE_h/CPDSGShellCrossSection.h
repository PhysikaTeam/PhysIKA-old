#pragma once
#include "baseshellsection.h"

class CPDSGShellCrossSection:public BaseShellCrossSection
{
public:
	virtual ~CPDSGShellCrossSection(){}

	__host__ __device__ virtual void computeInternalForce() override;
};