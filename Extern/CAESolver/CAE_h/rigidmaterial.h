#pragma once
#include"materialnew.h"

typedef struct RigidMaterial:MaterialNew
{
	__host__ __device__ RigidMaterial();

	__host__ __device__ RigidMaterial(Material *oldMat);

}RigidMaterial;