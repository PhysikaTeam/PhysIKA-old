#pragma once
#include"SurfaceInteract.h"

struct SurfaceInteract;

typedef struct ThermalSurfaceInteract:SurfaceInteract
{
	double gapConductance_;

	double gapClearance_;

	double averageTemperature_;

	double emissivityA_;

	double emissivityB_;

	double effectiveViewfactor_;

	__host__ __device__ ThermalSurfaceInteract();

	__host__ __device__ ThermalSurfaceInteract(ThermalSurfaceInteract *tm);

	__host__ void handleContactPairCpu(ContactPairCuda* cnPair, double dt) override;

	__device__ void handleContactPairGpu(ContactPairCuda* cnPair, double dt) override;

}ThermalSurfaceInteract;