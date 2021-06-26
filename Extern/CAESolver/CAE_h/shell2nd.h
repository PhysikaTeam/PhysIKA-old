#pragma once
#include"cuda_runtime.h"
#include"node.h"
#include"elementtypeenum.h"
#include"element.h"

struct NodeCuda;
struct Element;

typedef struct ShellElement2nd:Element
{

	int nodeIdArray_[8];
	NodeCuda* nodeArray_[8];

	int gpuId_;

	double dtEle_;

	double lengthChart_;

	double massScaleRate_;

	double volume_;

	double translateMass_;

	double rotateMass_[3];

	double fint[8][6];

	double thick_;
	double ini_thick_;

	double area;

	__host__ __device__ void setElementSectionType();

	//__host__ __device__ void computeAddMemSize();

	//__host__ __device__ void allocateAddMemory();
}ShellElement2nd;