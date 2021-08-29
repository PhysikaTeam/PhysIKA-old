#pragma once
#include"SurfaceInteract.h"

struct SurfaceInteract;

typedef struct DampSurfaceInteract:SurfaceInteract
{
	double dampCoeff_;

	double clearance_;

	double fraction_;

	__host__ __device__	DampSurfaceInteract();

	__host__ __device__ DampSurfaceInteract(DampSurfaceInteract *tm);

	__host__ void handleContactPairCpu(ContactPairCuda* cnPair, double dt) override;

	__device__ void handleContactPairGpu(ContactPairCuda* cnPair, double dt) override;

}DampSurfaceInteract;
