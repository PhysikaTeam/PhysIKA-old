#pragma once
#include"cuda_runtime.h"
#include"elementtypeenum.h"
#include"node.h"
#include "materialold.h"
#include"hourglasscontrol.h"
#include"element.h"

struct NodeCuda;
struct Element;

typedef struct BeamElement2nd:Element
{

	int nodeIdArray_[4];
	NodeCuda* nodeArray_[4];

	int gpuId_;

	double dtEle_;

	double lengthChart_;

	double massScaleRate_;

	double volume_;

	double translateMass_;

	double rotateMass_[3];

	double fint[4][6];

	__host__ __device__ void setElementSectionType();

	//__host__ __device__ void computeAddMemSize();

	//__host__ __device__ void allocateAddMemory();

}BeamElement2nd;