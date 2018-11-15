#pragma once

#include "Physika_Core/DataTypes.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "INeighbors.h"

namespace Physika{

#define INVALID -1
#define BUCKETS 8
#define CAPACITY 16

	template<typename TDataType>
	class GridHash
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

//		static_assert(Coord::dims() == 3, "GridHash only works for three dimensional spaces!");

		GridHash();
		~GridHash();

		void SetSpace(Real _h, Coord _lo, Coord _hi);

		void ConstructHashTable(Array<Coord>& pos);

		/*!
		*	\brief	May not be correct at extreme cases.
		*/
		void QueryNeighbors(Array<Coord>& posArr, Array<SPHNeighborList>& neighbors, Real h, Real pdist, int nbMaxNum);

		void QueryNeighborSlow(Array<Coord>& posArr, Array<SPHNeighborList>& neighbors, Real h, int nbMaxNum);

		void Clear();

		void Release();

		__device__ inline int GetIndex(int i, int j, int k)
		{
			if (i < 0 || i >= nx) return INVALID;
			if (j < 0 || j >= ny) return INVALID;
			if (k < 0 || k >= nz) return INVALID;

			return i + j*nx + k*nx*ny;
		}

		__device__ inline int GetIndex(Coord pos)
		{
			int i = floor((pos[0] - lo[0]) / ds);
			int j = floor((pos[1] - lo[1]) / ds);
			int k = floor((pos[2] - lo[2]) / ds);

			return GetIndex(i, j, k);
		}

		__device__ inline int3 GetIndex3(Coord pos)
		{
			int i = floor((pos[0] - lo[0]) / ds);
			int j = floor((pos[1] - lo[1]) / ds);
			int k = floor((pos[2] - lo[2]) / ds);

			return make_int3(i, j, k);
		}

		__device__ inline int GetCounter(int gId) { return counter[gId]; }

		__device__ inline int GetParticleId(int gId, int n) { return ids[gId*npMax + n]; }

	public:
		int num;
		int nx, ny, nz;

		Real ds;

		Coord lo;
		Coord hi;

		int npMax;		//maximum particle number for each cell

		int* ids;
		int* counter;
	};

#ifdef PRECISION_FLOAT
	template class GridHash<DataType3f>;
#else
	template class GridHash<DataType3d>;
#endif
}
