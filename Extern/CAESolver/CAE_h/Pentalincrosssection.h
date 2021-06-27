#pragma once

#include "basesolidcrosssection.h"

class PentaLinCrossSection:public BaseSolidCrossSection
{
	__host__ __device__ virtual void computeInternalForce() override;

	__host__ __device__ virtual void computeHourglassForce() override;
};