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
	void GridHash<TDataType>::SetSpace(Real _h, Coord _lo, Coord _hi)
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
		if (pId >= pos.Size()) return;

		//hash.PushPosition(pos[pId], pId);

		int gId = hash.GetIndex(pos[pId]);

		if (gId < 0) return;

		int index = atomicAdd(&(hash.counter[gId]), 1);
		index = index < hash.npMax - 1 ? index : hash.npMax - 1;
		hash.ids[gId * hash.npMax + index] = pId;
	}

	template<typename TDataType>
	void GridHash<TDataType>::ConstructHashTable(Array<Coord>& pos)
	{
		dim3 pDims = int(ceil(pos.Size() / BLOCK_SIZE + 0.5f));
		K_ConstructHashTable << <pDims, BLOCK_SIZE >> > (*this, pos);
	}

	template<typename TDataType>
	void GridHash<TDataType>::Clear()
	{
		cudaCheck(cudaMemset(counter, 0, num * sizeof(int)));
	}

	template<typename TDataType>
	void GridHash<TDataType>::Release()
	{
		cudaCheck(cudaFree(counter));
		cudaCheck(cudaFree(ids));
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ComputeNeighbors(
		Physika::Array<Coord> posArr, 
		Physika::Array<NeighborList> neighbors, 
		Physika::GridHash<TDataType> hash, 
		Real h, 
		Real pdist, 
		int nbMaxNum)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId > posArr.Size()) return;

		int tId = threadIdx.x;
		int ids[BUCKETS][CAPACITY];
		Real distance[CAPACITY];
		int counter[BUCKETS];

		for (int i = 0; i < BUCKETS; i++)
		{
			counter[i] = 0;
		}

		Coord pos_ijk = posArr[pId];
		int3 gId3 = hash.GetIndex3(pos_ijk);

		for (int c = 0; c < 27; c++)
		{
			int cId = hash.GetIndex(gId3.x + offset[c][0], gId3.y + offset[c][1], gId3.z + offset[c][2]);
			if (cId >= 0) {
				int totalNum = min(hash.GetCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.GetParticleId(cId, i);
					Real d_ij = (pos_ijk - posArr[nbId]).norm();
					if (d_ij < h)
					{
						int bId = floor(pow(d_ij / h, Real(3))*BUCKETS);
						bId = clamp(bId, 0, BUCKETS - 1);
//						printf("exceeded %i", bId);
						if (counter[bId] < CAPACITY)
						{
							ids[bId][counter[bId]] = nbId;
							counter[bId]++;
						}
// 						else
// 						{
// 							printf("exceeded");
// 						}
					}
				}
			}
		}

		int nbSize = 0;
		int totalNum = 0;
		int bId;
		for (bId = 0; bId < BUCKETS; bId++)
		{
			int btSize = counter[bId];
			totalNum += btSize;
			if (totalNum <= nbMaxNum)
			{
				for (int k = 0; k < btSize; k++)
				{
					neighbors[pId][nbSize] = ids[bId][k];
					nbSize++;
				}
			}
			else
			{
				for (int i = 0; i < btSize; i++)
				{
					distance[i] = (pos_ijk - posArr[ids[bId][i]]).norm();
				}
				int rN = nbMaxNum - totalNum + btSize;
				for (int k = 0; k < rN; k++)
				{
					Real minDist = distance[k];
					int id = k;
					for (int t = k + 1; t < btSize; t++)
					{
						if (distance[t] < minDist)
						{
							minDist = distance[t];
							id = t;
						}
					}
					neighbors[pId][nbSize] = ids[bId][id];
					nbSize++;
					distance[id] = distance[k];
					ids[bId][id] = ids[bId][k];
				}
			}
		}

		neighbors[pId].size = nbSize;
	}

	template<typename TDataType>
	void GridHash<TDataType>::QueryNeighbors(Array<Coord>& posArr, Array<NeighborList>& neighbors, Real h, Real pdist, int nbMaxNum)
	{
		Clear();
		ConstructHashTable(posArr);

		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));

		K_ComputeNeighbors <Real, Coord> << <pDims, BLOCK_SIZE >> >(posArr, neighbors, *this, h, pdist, nbMaxNum);
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ComputeNeighborSlow(Physika::Array<Coord> posArr, Physika::Array<NeighborList> neighbors, Physika::GridHash<TDataType> hash, Real h, int nbMaxNum)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId > posArr.Size()) return;

		int tId = threadIdx.x;
		int ids[NEIGHBOR_SIZE];
		Real distance[NEIGHBOR_SIZE];

		Coord pos_ijk = posArr[pId];
		int3 gId3 = hash.GetIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.GetIndex(gId3.x + offset[c][0], gId3.y + offset[c][1], gId3.z + offset[c][2]);
			if (cId >= 0) {
				int totalNum = min(hash.GetCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.GetParticleId(cId, i);
					float d_ij = (pos_ijk - posArr[nbId]).norm();
					if (d_ij < h)
					{
						if (counter < nbMaxNum)
						{
							ids[counter] = nbId;
							distance[counter] = d_ij;
							counter++;
						}
						else
						{
							int maxId = 0;
							float maxDist = distance[0];
							for (int ne = 1; ne < nbMaxNum; ne++)
							{
								if (maxDist < distance[ne])
								{
									maxDist = distance[ne];
									maxId = ne;
								}
							}
							if (d_ij < distance[maxId])
							{
								distance[maxId] = d_ij;
								ids[maxId] = nbId;
							}
						}
					}
				}
			}
		}

		int bId;
		for (bId = 0; bId < counter; bId++)
		{
			neighbors[pId][bId] = ids[bId];
		}

		neighbors[pId].size = counter;
	}

	template<typename TDataType>
	void GridHash<TDataType>::QueryNeighborSlow(Array<Coord>& posArr, Array<NeighborList>& neighbors, Real h, int nbMaxNum)
	{
		Clear();
		ConstructHashTable(posArr);

		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));

// 		CTimer timer;
// 		timer.Start();
		K_ComputeNeighborSlow << <pDims, BLOCK_SIZE >> > (posArr, neighbors, *this, h, nbMaxNum);

// 		timer.Stop();
// 		timer.OutputString("Construction");
	}

}