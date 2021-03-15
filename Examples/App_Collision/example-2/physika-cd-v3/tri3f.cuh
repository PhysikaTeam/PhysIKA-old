#pragma once

typedef unsigned int uint;
#define MAX_PAIR_NUM 40000000

typedef struct {
	uint3 _ids;

	inline __device__ __host__ uint id0() const { return _ids.x; }
	inline __device__ __host__ uint id1() const { return _ids.y; }
	inline __device__ __host__ uint id2() const { return _ids.z; }
	inline __device__ __host__ uint id(int i) const { return (i == 0 ? id0() : ((i == 1) ? id1() : id2())); }
} tri3f;


typedef struct {
	uint4 _ids;

	inline __device__ __host__ uint id0() const { return _ids.x; }
	inline __device__ __host__ uint id1() const { return _ids.y; }
	inline __device__ __host__ uint id2() const { return _ids.z; }
	inline __device__ __host__ uint id3() const { return _ids.w; }
	inline __device__ __host__ uint id(int i) const {
		return (i == 3 ? id3() : (i == 0 ? id0() : ((i == 1) ? id1() : id2())));
	}
} tri4f;

inline __device__ bool covertex(int tA, int tB, tri3f *Atris)
{
	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++) {
			if (Atris[tA].id(i) == Atris[tB].id(j))
				return true;
		}

	return false;
}

inline __device__ int addPair(uint a, uint b, int2 *pairs, uint *idx)
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
	uint isVF, uint id1, uint id2, uint id3, uint id4, REAL d,int* t,
	int2 *pairs, int4 *dv, int *VF_EE, REAL *dt,int *CCDres, uint *idx)
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

		CCDres[offset]= (t==NULL) ? 0 : t[0];

		return offset;
	}

	return -1;
}

