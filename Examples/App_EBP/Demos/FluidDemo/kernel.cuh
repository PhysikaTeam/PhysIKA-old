#include "helper_math.h"
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//#define PI (float)3.14159265358979323846264338327950288
//// ºËº¯ÊýPoly6
//// p1, p2: positions of particle, coreRadius: core radius of kernel function
//__host__ __device__  float poly6(float3 &p1, float3 &p2, double coreRadius)
//{
//	float3 distance;
//	float result;
//	distance.x = p1.x - p2.x;
//	distance.y = p1.y - p2.y;
//	distance.z = p1.z - p2.z;
//	
//	float rl = length(distance);
//	
//	float h9 = pow(coreRadius, 9);
//	float m_k = 315 / (64 * PI*h9);
//	if (rl >= 0 && rl <= coreRadius)
//	{
//		float h2 = coreRadius * coreRadius;
//		float r2 = rl*rl;
//	    float hr = h2 - r2;
//		result = m_k*hr*hr*hr;
//	}
//	else
//	{
//		result = 0;
//	}
//
//	return result;
//}