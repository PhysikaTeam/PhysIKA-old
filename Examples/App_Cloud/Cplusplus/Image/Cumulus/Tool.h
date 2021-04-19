#pragma once
#include "Vector.h"
#include "global.h"
class Tool {
public:
	float PhaseFunction(Vector3 v1, Vector3 v2);
	float PhaseFunction(float  cosAngle);
	bool isProbabilityGreater(float threahold);
	float Rand(float vaue1, float value2);
	void ComputeTriangleNormal(float normal[3], float PA[3], float PB[3], float PC[3]);
	Vector3   MatMult(float Mat[9], Vector3 vec);
	void PrintRunnngIfo(char* ifo);
};