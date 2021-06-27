#pragma once
#include"baseshellsection.h"

class Mem3NodeCrossSection:public BaseShellCrossSection
{
	__host__ __device__ virtual void computeInternalForce() override;
};