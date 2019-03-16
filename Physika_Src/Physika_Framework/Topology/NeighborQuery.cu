#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Framework/Framework/Node.h"
#include "NeighborQuery.h"
#include "Physika_Framework/Topology/NeighborList.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

namespace Physika
{
	__constant__ int offset1[27][3] = { 0, 0, 0,
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
	NeighborQuery<TDataType>::NeighborQuery()
		: ComputeModule()
		, m_posID(MechanicalState::position())
		, m_adaptNID(MechanicalState::particle_neighbors())
		, m_radius(Real(0.01))
		, m_maxNum(30)
	{
		Coord lowerBound(0);
		Coord upperBound(1.0);
		hash.setSpace(m_radius, lowerBound, upperBound);
	}


	template<typename TDataType>
	NeighborQuery<TDataType>::NeighborQuery(Real s, Coord lo, Coord hi)
		: ComputeModule()
		, m_posID(MechanicalState::position())
		, m_adaptNID(MechanicalState::particle_neighbors())
		, m_radius(s)
		, m_maxNum(30)
	{
		hash.setSpace(m_radius, lo, hi);
	}


	template<typename TDataType>
	void NeighborQuery<TDataType>::compute()
	{
		auto mstate = getParent()->getMechanicalState();
		auto nbrFd = mstate->getField<NeighborField<int>>(m_adaptNID);
		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);

		queryParticleNeighbors(nbrFd->getValue(), posFd->getValue(), m_radius);
	}


	template<typename TDataType>
	void NeighborQuery<TDataType>::queryParticleNeighbors(NeighborList<int>& nbr, DeviceArray<Coord>& pos, Real radius)
	{
		hash.clear();
		hash.construct(pos);

		if (!nbr.isLimited())
		{
			queryNeighborDynamic(nbr, pos, radius);
		}
		else
		{
			queryNeighborFixed(nbr, pos, radius);
		}
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_CalNeighborSize(DeviceArray<int> count, DeviceArray<Coord> posArr, GridHash<TDataType> hash, Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId > posArr.size()) return;

		Coord pos_ijk = posArr[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
			if (cId >= 0) {
				int totalNum = min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					Real d_ij = (pos_ijk - posArr[nbId]).norm();
					if (d_ij < h)
					{
						counter++;
					}
				}
			}
		}

		count[pId] = counter;
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_GetNeighborElements(NeighborList<int> nbr, DeviceArray<Coord> posArr, GridHash<TDataType> hash, Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId > posArr.size()) return;

		Coord pos_ijk = posArr[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int j = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
			if (cId >= 0) {
				int totalNum = min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					Real d_ij = (pos_ijk - posArr[nbId]).norm();
					if (d_ij < h)
					{
						nbr.setElement(pId, j, nbId);
						j++;
					}
				}
			}
		}
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryNeighborSize(DeviceArray<int>& num, DeviceArray<Coord>& pos, Real h)
	{
		uint pDims = cudaGridSize(pos.size(), BLOCK_SIZE);
		K_CalNeighborSize << <pDims, BLOCK_SIZE >> > (num, pos, hash, h);
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryNeighborDynamic(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h)
	{
		DeviceArray<int>& nbrNum = nbrList.getIndex();

		queryNeighborSize(nbrNum, pos, h);

		int sum = thrust::reduce(thrust::device, nbrNum.getDataPtr(), nbrNum.getDataPtr()+ nbrNum.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, nbrNum.getDataPtr(), nbrNum.getDataPtr() + nbrNum.size(), nbrNum.getDataPtr());

		DeviceArray<int>& elements = nbrList.getElements();
		elements.resize(sum);

		uint pDims = cudaGridSize(pos.size(), BLOCK_SIZE);
		K_GetNeighborElements << <pDims, BLOCK_SIZE >> > (nbrList, pos, hash, h);
	}


/*	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ComputeNeighbors(
		Physika::Array<Coord> posArr,
		Physika::Array<SPHNeighborList> neighbors,
		Physika::GridHash<TDataType> hash,
		Real h,
		Real pdist,
		int nbMaxNum)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId > posArr.size()) return;

		int tId = threadIdx.x;
		int ids[BUCKETS][CAPACITY];
		Real distance[CAPACITY];
		int counter[BUCKETS];

		for (int i = 0; i < BUCKETS; i++)
		{
			counter[i] = 0;
		}

		Coord pos_ijk = posArr[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset[c][0], gId3.y + offset[c][1], gId3.z + offset[c][2]);
			if (cId >= 0) {
				int totalNum = min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
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
	}*/

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ComputeNeighborFixed(
		NeighborList<int> neighbors, 
		DeviceArray<Coord> posArr, 
		GridHash<TDataType> hash, 
		Real h,
		int* heapIDs,
		Real* heapDistance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId > posArr.size()) return;

		int nbrLimit = neighbors.getNeighborLimit();

		int* ids(heapIDs + pId * nbrLimit);// = new int[nbrLimit];
		Real* distance(heapDistance + pId * nbrLimit);// = new Real[nbrLimit];

		Coord pos_ijk = posArr[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset1[c][0], gId3.y + offset1[c][1], gId3.z + offset1[c][2]);
			if (cId >= 0) {
				int totalNum = min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					float d_ij = (pos_ijk - posArr[nbId]).norm();
					if (d_ij < h)
					{
						if (counter < nbrLimit)
						{
							ids[counter] = nbId;
							distance[counter] = d_ij;
							counter++;
						}
						else
						{
							int maxId = 0;
							float maxDist = distance[0];
							for (int ne = 1; ne < nbrLimit; ne++)
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

		neighbors.setNeighborLimit(pId, counter);

		int bId;
		for (bId = 0; bId < counter; bId++)
		{
			neighbors.setElement(pId, bId, ids[bId]);
		}

// 		delete[] ids;
// 		delete[] distance;
	}

	template<typename TDataType>
	void NeighborQuery<TDataType>::queryNeighborFixed(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h)
	{
		int num = pos.size();
		int* ids;
		Real* distance;
		cudaMalloc((void**)&ids, num * sizeof(int) * nbrList.getNeighborLimit());
		cudaMalloc((void**)&distance, num * sizeof(int) * nbrList.getNeighborLimit());

		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		K_ComputeNeighborFixed << <pDims, BLOCK_SIZE >> > (nbrList, pos, hash, h, ids, distance);

		cudaFree(ids);
		cudaFree(distance);
	}
}