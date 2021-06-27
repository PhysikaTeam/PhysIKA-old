#pragma once
#include<string>
#include"contactpair.h"

using std::string;

struct ContactPairCuda;

enum SurfaceInteractType
{
	Friction,
	Thermal,
	Damp,
};

typedef struct SurfaceInteract
{
	int id;

	SurfaceInteractType type;

	__host__ __device__ SurfaceInteract();

	__host__  virtual void handleContactPairCpu(ContactPairCuda *cnPair, double dt) = 0;

	__device__ virtual void handleContactPairGpu(ContactPairCuda *cnPair, double dt)=0;

	__host__ __device__ virtual ~SurfaceInteract();
}SurfaceInteract;