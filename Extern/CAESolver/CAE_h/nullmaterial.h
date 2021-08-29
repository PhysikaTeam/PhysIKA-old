#pragma once
#include"materialnew.h"

typedef struct NullMaterial:MaterialNew
{
	__host__ __device__ NullMaterial();

	__host__ __device__ NullMaterial(Material *oldMat);

}NullMaterial;