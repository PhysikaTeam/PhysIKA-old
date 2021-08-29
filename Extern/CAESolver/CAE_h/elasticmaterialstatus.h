#pragma once
#include"materialstatus.h"

struct ElasticMaterialStatus:MaterialStatus
{
	__host__ __device__ ElasticMaterialStatus() :MaterialStatus() { ; }
};