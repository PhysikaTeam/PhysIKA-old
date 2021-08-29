#pragma once
#include "materialstatus.h"
#include"elementtypeenum.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

struct Element;
struct MaterialStatus;

typedef struct GaussPoint
{
	int nodeNumber_;

	int matStatusStoreId_;

	double weight_;
	
	double3 naturalCoord_;

	MaterialStatus* materialStatus_;

	Element* element_;

	ElementNewType elementType_;

	double* shapePartialDx_;
	double* shapePartialDy_;
	double* shapePartialDz_;

	double domainVolme;

public:

	GaussPoint(int n, double3 co, double w, Element *el);

	GaussPoint();

	__host__ __device__ double giveWeight() { return weight_; }

	__host__ __device__ void setWeigth(double w) { weight_ = w; }

	__host__ __device__ double3 giveNaturalCoord() { return naturalCoord_; }

	__host__ __device__ void setNaturalCoord(double3 co) { naturalCoord_ = co; }

	__host__ __device__ void setNaturalCoord(double co1, double co2 = 0, double co3 = 0)
	{
		naturalCoord_.x = co1, naturalCoord_.y = co2, naturalCoord_.z = co3;
	}

	__host__ __device__ Element* giveElement() { return element_; }

	__host__ __device__ void setMatStatusPtr(MaterialStatus *matStatusPtr);

	__host__ __device__ void setNodeNumAndAllocateShapePartial();

	__host__ __device__ virtual ~GaussPoint();
}GaussPoint;