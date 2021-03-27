#ifndef _pbf_solver_kernel_W_
#define _pbf_solver_kernel_W_

#include "helper_math.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define PI (float)3.14159265358979323846264338327950288
// �˺���Poly6
// p1, p2: positions of particle, coreRadius: core radius of kernel function
__host__ __device__  double poly6(float3 &p1, float3 &p2, double coreRadius)
{
	float3 distance;
	double result;
	distance.x = p1.x - p2.x;
	distance.y = p1.y - p2.y;
	distance.z = p1.z - p2.z;
	
	float rl = length(distance);
	//float rl = r / coreRadius;
	float h9 = pow(coreRadius, 9);
	float m_k = 315.0f / (64 * PI * h9);
	if (rl >= 0 && rl <= coreRadius)
	{
		float h2 = coreRadius * coreRadius;
		float r2 = rl*rl;
		float hr = h2 - r2;
		result = m_k*hr*hr*hr;
	}
	else
	{
		result = 0;
	}
	return result;
}

//gradient of kernel function poly6
__host__ __device__ float3 grad_poly6(float3 &p1, float3 &p2, double coreRadius)
{
	float3 res;
	double h9 = pow(coreRadius, 9);
	double m_k;
	float3 distance;
	distance.x = p1.x - p2.x;
	distance.y = p1.y - p2.y;
	distance.z = p1.z - p2.z;

	m_k = 945.0 / (32 * PI * h9);
	double rl = length(distance);
	if (rl > 0)
	{
		float3 r = normalize(distance);
		if (rl > 0 && rl <= coreRadius)
		{
			double h2 = coreRadius * coreRadius;
			double r2 = rl*rl;
			double hr = h2 - r2;
			float value = -1 * m_k * hr * hr;
			res = make_float3(value*r.x, value*r.y, value*r.z);
		}
	}
	else
	{
		res = make_float3(0, 0, 0);
	}
	
	return res;
}

//Poly6�˺�����Laplacian
__host__ __device__ double laplacian_poly6(float3 &p1, float3 &p2, float coreRadius)
{
	double res;
	double h9 = pow(coreRadius, 9);
	double m_k;
	m_k = 945.0 / (8 * PI*h9);
	float3 distance;
	distance.x = p1.x - p2.x;
	distance.y = p1.y - p2.y;
	distance.z = p1.z - p2.z;

	double rl = length(distance);
	if (rl >= 0 && rl <= coreRadius)
	{
		double h2 = coreRadius * coreRadius;
		double r2 = rl*rl;
		double hr = h2 - r2;
		res = m_k*(hr*hr)*(r2 - 0.75*hr);
	}
	else
	{
		res = 0;
	}
	return res;
}


__host__ __device__  double poly6_h(float h, double coreRadius)
{
	float3 distance;
	double result;

	float h9 = pow(coreRadius, 9);
	float m_k = 315 / (64 * PI*h9);
	if (h >= 0 && h <= coreRadius)
	{
		float h2 = coreRadius * coreRadius;
		float r2 = h*h;
		float hr = h2 - r2;
		result = m_k*hr*hr*hr;
	}
	else
	{
		result = 0;
	}
	return result;
}

__host__ __device__ float3 grad_spiky(float3 p1, float3 p2, double coreRadius)
{
	float3 r = make_float3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
	float r_len = length(r);
	if (r_len <= 0.0 || r_len > coreRadius) return make_float3(0.0f,0.0f,0.0f);

	float x = (coreRadius - r_len) / pow(coreRadius, 3);
	float g_factor = (-45.0 / PI)*x*x;
	float3 result = normalize(r)*g_factor;
	return result;
}

//����h=1
__host__ __device__ float3 grad_spiky_new(float3 p1, float3 p2, double coreRadius)
{
	float3 r = make_float3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
	float r_len = length(r);
	if (r_len <= 0.0 || r_len > coreRadius) return make_float3(0.0f, 0.0f, 0.0f);

	float h = 1;
	float r_len_new = r_len / coreRadius;
	float x = (h - r_len_new) / pow(h, 3);
	float g_factor = (-45.0 / PI)*x*x;
	float3 result = normalize(r)*g_factor;
	return result;
}

__host__ __device__  double poly6_new(float3 &p1, float3 &p2, double coreRadius)
{
	float3 distance;
	double result;
	distance.x = p1.x - p2.x;
	distance.y = p1.y - p2.y;
	distance.z = p1.z - p2.z;

	float rl = length(distance);
	float rl_new = rl / coreRadius;

	float  h = 1;
	float h9 = pow(h, 9);
	float m_k = 315.0f / (64 * PI * h9);
	if (rl >= 0 && rl <= coreRadius)
	{
		float h2 = h * h;
		float r2 = rl_new*rl_new;
		float hr = h2 - r2;
		result = m_k*hr*hr*hr;
	}
	else
	{
		result = 0;
	}
	return result;
}


//
__host__ __device__  float vectroLen(float3 r)
{
	float result;
	result = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
	return result;
}
__host__ __device__  double W(float3 &p1, float3 &p2, double coreRadius)
{
	double h3 = coreRadius*coreRadius*coreRadius;
	float m_k;
	m_k = 8.0 / (PI*h3);
	
	float3 r;
	float result = 0.0;
	r.x = p1.x - p2.x;
	r.y = p1.y - p2.y;
	r.z = p1.z - p2.z;

	float rl = vectroLen(r);
	float q = rl / coreRadius;
	if (q <= 1.0)
	{
		if (q <= 0.5)
		{
			float q2 = q*q;
			float q3 = q2*q;
			result = m_k*(6.0*q3 - 6.0*q2 + 1.0);
		}
		else
		{
			result = m_k*(2.0*pow(1.0 - q, 3));
		}
	}
	return result;
}

__host__ __device__ float3 gradW(float3 p1, float3 p2, double coreRadius)
{
	float h3 = coreRadius*coreRadius*coreRadius;
	float m_l = 48.0 / (PI*h3);

	float3 result;
	float3 r;
	r.x = p1.x - p2.x;
	r.y = p1.y - p2.y;
	r.z = p1.z - p2.z;
	
	float rl = vectroLen(r);
	float q = rl / coreRadius;
	if (q <= 1.0)
	{
		if (rl > 1.0e-6)
		{
			float3 gradq = r*(1.0 / (rl*coreRadius));
			if (q <= 0.5)
			{
				result = m_l*q*(3.0*q - 2.0)*gradq;
			}
			else
			{
				float factor = 1.0 - q;
				result = m_l*(-factor*factor)*gradq;
			}
		}
	}
	else
	{
		result = make_float3(0.0, 0.0, 0.0);
	}
	return result;
}

#endif 