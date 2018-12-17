#include "GridHash.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"

namespace Physika{

	__constant__ int offset[27][3] = { 0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 0, -1,
		0, -1, 0,
		-1, 0, 0,
		0, 1, 1,
		0, 1, -1,
		0, -1, 1,
		0, -1, -1,
		1, 0, 1,
		1, 0, -1,
		-1, 0, 1,
		-1, 0, -1,
		1, 1, 0,
		1, -1, 0,
		-1, 1, 0,
		-1, -1, 0,
		1, 1, 1,
		1, 1, -1,
		1, -1, 1,
		-1, 1, 1,
		1, -1, -1,
		-1, 1, -1,
		-1, -1, 1,
		-1, -1, -1
	};

	template<typename TDataType>
	GridHash<TDataType>::GridHash()
	{
	}

	template<typename TDataType>
	GridHash<TDataType>::~GridHash()
	{
	}

	template<typename TDataType>
	void GridHash<TDataType>::setSpace(Real _h, Coord _lo, Coord _hi)
	{
		int padding = 2;
		ds = _h;
		lo = _lo- padding*ds;

		Coord nSeg = (_hi - _lo) / ds;

		nx = ceil(nSeg[0]) + 1 + 2 * padding;
		ny = ceil(nSeg[1]) + 1 + 2 * padding;
		nz = ceil(nSeg[2]) + 1 + 2 * padding;
		hi = lo + Coord(nx, ny, nz)*ds;

		num = nx*ny*nz;

		npMax = 32;

		cudaCheck(cudaMalloc((void**)&counter, num * sizeof(int)));
		cudaCheck(cudaMalloc((void**)&ids, num * npMax * sizeof(int)));
	}

	template<typename TDataType>
	__global__ void K_ConstructHashTable(GridHash<TDataType> hash, Array<TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId < 0) return;

		int index = atomicAdd(&(hash.counter[gId]), 1);
		index = index < hash.npMax - 1 ? index : hash.npMax - 1;
		hash.ids[gId * hash.npMax + index] = pId;
	}

	template<typename TDataType>
	void GridHash<TDataType>::construct(DeviceArray<Coord>& pos)
	{
		dim3 pDims = int(ceil(pos.size() / BLOCK_SIZE + 0.5f));
		K_ConstructHashTable << <pDims, BLOCK_SIZE >> > (*this, pos);
	}

	template<typename TDataType>
	void GridHash<TDataType>::clear()
	{
		cudaCheck(cudaMemset(counter, 0, num * sizeof(int)));
	}

	template<typename TDataType>
	void GridHash<TDataType>::release()
	{
		cudaCheck(cudaFree(counter));
		cudaCheck(cudaFree(ids));
	}
}