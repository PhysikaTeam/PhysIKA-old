#pragma once

#include "Framework/DataTypes.h"
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

		GridHash();
		~GridHash();

		void SetSpace(Real _h, Coord _lo, Coord _hi);

		void ConstructHashTable(Array<Coord>& pos);

		/*!
		*	\brief	May not be correct at extreme cases.
		*/
		void QueryNeighbors(Array<Coord>& posArr, Array<NeighborList>& neighbors, Real h, Real pdist, int nbMaxNum);

		void QueryNeighborSlow(Array<Coord>& posArr, Array<NeighborList>& neighbors, Real h, int nbMaxNum);

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
			int i = floor((pos.x - lo.x) / ds);
			int j = floor((pos.y - lo.y) / ds);
			int k = floor((pos.z - lo.z) / ds);

			return GetIndex(i, j, k);
		}

		__device__ inline int3 GetIndex3(Coord pos)
		{
			int i = floor((pos.x - lo.x) / ds);
			int j = floor((pos.y - lo.y) / ds);
			int k = floor((pos.z - lo.z) / ds);

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

	template class GridHash<DataType3f>;
}
