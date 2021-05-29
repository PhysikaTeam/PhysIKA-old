#pragma once
#include <cuda_runtime.h>
#include "Core/Utility/cuda_helper_math.h"

// --------------------- from tri3f.cuh ----------------------
typedef unsigned int uint;
#define MAX_PAIR_NUM 40000000

typedef struct {
	uint3 _ids;

	inline __device__ __host__ uint id0() const { return _ids.x; }
	inline __device__ __host__ uint id1() const { return _ids.y; }
	inline __device__ __host__ uint id2() const { return _ids.z; }
	inline __device__ __host__ uint id(int i) const { return (i == 0 ? id0() : ((i == 1) ? id1() : id2())); }
} tri3f;

inline __device__ bool covertex(int tA, int tB, tri3f* Atris)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			if (Atris[tA].id(i) == Atris[tB].id(j))
				return true;
		}

	return false;
}

inline __device__ int addPair(uint a, uint b, int2* pairs, uint* idx)
{
	if (*idx < MAX_PAIR_NUM)
	{
		uint offset = atomicAdd(idx, 1);
		pairs[offset].x = a;
		pairs[offset].y = b;

		return offset;
	}

	return -1;
}

//isVF  0:VF  1:EE
inline __device__ int addPairDCD(uint a, uint b,
	uint isVF, uint id1, uint id2, uint id3, uint id4, double d, int* t,
	int2* pairs, int4* dv, int* VF_EE, double* dt, int* CCDres, uint* idx)
{
	if (*idx < MAX_PAIR_NUM)
	{
		uint offset = atomicAdd(idx, 1);
		pairs[offset].x = a;
		pairs[offset].y = b;

		if (VF_EE)
			VF_EE[offset] = isVF;

		if (dv) {
			dv[offset].x = id1;
			dv[offset].y = id2;
			dv[offset].z = id3;
			dv[offset].w = id4;
		}

		if (dt)
			dt[offset] = d;

		CCDres[offset] = (t == NULL) ? 0 : t[0];

		return offset;
	}

	return -1;
}

// --------------------- from tri-contact.cuh ----------------------
#pragma once

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

inline __device__ int project3(const double3& ax,
	const double3& p1, const double3& p2, const double3& p3)
{
	double P1 = dot(ax, p1);
	double P2 = dot(ax, p2);
	double P3 = dot(ax, p3);

	double mx1 = fmaxf(P1, fmaxf(P2, P3));
	double mn1 = fminf(P1, fminf(P2, P3));

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;

	return 1;
}

inline  __device__ int project6(double3& ax,
	double3& p1, double3& p2, double3& p3,
	double3& q1, double3& q2, double3& q3)
{
	double P1 = dot(ax, p1);
	double P2 = dot(ax, p2);
	double P3 = dot(ax, p3);
	double Q1 = dot(ax, q1);
	double Q2 = dot(ax, q2);
	double Q3 = dot(ax, q3);

	double mx1 = fmaxf(P1, fmaxf(P2, P3));
	double mn1 = fminf(P1, fminf(P2, P3));
	double mx2 = fmaxf(Q1, fmaxf(Q2, Q3));
	double mn2 = fminf(Q1, fminf(Q2, Q3));

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;

	return 1;
}

inline __device__ bool
tri_contact(double3& P1, double3& P2, double3& P3,
	double3& Q1, double3& Q2, double3& Q3)
{
	double3 p1 = zero3f();;
	double3 p2 = P2 - P1;
	double3 p3 = P3 - P1;
	double3 q1 = Q1 - P1;
	double3 q2 = Q2 - P1;
	double3 q3 = Q3 - P1;

	double3 e1 = p2 - p1;
	double3 e2 = p3 - p2;
	double3 e3 = p1 - p3;

	double3 f1 = q2 - q1;
	double3 f2 = q3 - q2;
	double3 f3 = q1 - q3;

	double3 n1 = 
		e1, e2);
	double3 m1 = cross(f1, f2);

	double3 g1 = cross(e1, n1);
	double3 g2 = cross(e2, n1);
	double3 g3 = cross(e3, n1);

	double3 h1 = cross(f1, m1);
	double3 h2 = cross(f2, m1);
	double3 h3 = cross(f3, m1);

	double3 ef11 = cross(e1, f1);
	double3 ef12 = cross(e1, f2);
	double3 ef13 = cross(e1, f3);
	double3 ef21 = cross(e2, f1);
	double3 ef22 = cross(e2, f2);
	double3 ef23 = cross(e2, f3);
	double3 ef31 = cross(e3, f1);
	double3 ef32 = cross(e3, f2);
	double3 ef33 = cross(e3, f3);

	// now begin the series of tests
	if (!project3(n1, q1, q2, q3)) return false;
	if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
	if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

	return true;
}