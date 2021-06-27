#pragma once
#include"SurfaceInteract.h"

struct SurfaceInteract;

typedef struct FrictionSurfaceInteract :SurfaceInteract
{
	bool isAnisotropic_;

	double statFrictionCoeff_;

	double dynaFrictionCoeff_;

	double deCayCoeff_;

	double slipRate_;

	double contactPressure_;

	double averageTemperature_;

	//abaqusÖÐµÄrough friction
	bool isRough_;

	__host__ __device__ FrictionSurfaceInteract();

	__host__ virtual void handleContactPairCpu(ContactPairCuda* cnPair, double dt) override;

	__device__ virtual void handleContactPairGpu(ContactPairCuda* cnPair, double dt) override;

	__host__ __device__ FrictionSurfaceInteract(FrictionSurfaceInteract *tm);

}FrictionSurfaceInteract;