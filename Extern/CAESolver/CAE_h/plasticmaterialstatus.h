#pragma once
#include"elasticmaterialstatus.h"

struct PlasticMaterialStatus:public ElasticMaterialStatus
{
	double yieldStress_;

	double effectiveStrain_;

	bool loadFlag_;

	__host__ __device__
	PlasticMaterialStatus(): ElasticMaterialStatus(), yieldStress_(0), effectiveStrain_(0), loadFlag_(false)
	{
	}
};