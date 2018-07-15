#include "DensitySimple.h"

namespace Physika
{
// 	struct DIS_STATE
// 	{
// 		Real samplingDistance;
// 		Real smoothingLength;
// 		SpikyKernel kernSpiky;
// 	};

//	__constant__ DIS_STATE const_dis_state;

	template<typename Real, typename Coord>
	__global__ void DIS_ComputeDelta(
		DeviceArray<Coord> dPos, 
		DeviceArray<Coord> posArr, 
		DeviceArray<NeighborList> neighbors,
		Real smoothingLength,
		Real samplingDistance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		Coord pos_i = posArr[pId];

		Coord deltaX(0);
		Real total_weight = 0;

		SpikyKernel<Real> kern;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r < samplingDistance && r > EPSILON)
			{
				Real weight_ij = kern.Weight(r, smoothingLength);
				deltaX = weight_ij * (samplingDistance -r) * (pos_i - posArr[j]) * (1.0f / r);

				total_weight += weight_ij;
			}
		}

		if (total_weight < EPSILON)
		{
			total_weight = 1.0f;
		}

		Coord dP_ij;

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r < samplingDistance && r > EPSILON)
			{
				Real weight_ij = kern.Weight(r, smoothingLength) / total_weight;
				dP_ij = 0.2f*weight_ij * (samplingDistance - r) * (pos_i - posArr[j]) * (1.0f / r);

				atomicAdd(&dPos[pId][0], dP_ij[0]);
				atomicAdd(&dPos[j][0], -dP_ij[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dP_ij[1]);
					atomicAdd(&dPos[j][1], -dP_ij[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dP_ij[2]);
					atomicAdd(&dPos[j][2], -dP_ij[2]);
				}
// 
// 				atomicAdd(&dPos[pId].x, dP_ij.x);
// 				atomicAdd(&dPos[pId].y, dP_ij.y);
// 				atomicAdd(&dPos[pId].z, dP_ij.z);
// 				atomicAdd(&dPos[j].x, -dP_ij.x);
// 				atomicAdd(&dPos[j].y, -dP_ij.y);
// 				atomicAdd(&dPos[j].z, -dP_ij.z);
			}
		}
	}

	template<typename Real, typename Coord>
	__global__ void DIS_UpdatePosition(
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dPos, 
		DeviceArray<Attribute> attArr, 
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;
		if (!attArr[pId].IsDynamic()) return;

		posArr[pId] += dPos[pId];
		velArr[pId] += dPos[pId] / dt;
	}

	template<typename TDataType>
	DensitySimple<TDataType>::DensitySimple(
		ParticleSystem<TDataType>* parent)
		: DensityConstraint(parent)
	{
		assert(m_parent != NULL);

		setInputSize(1);
		setOutputSize(1);

		int num = m_parent->GetParticleNumber();

		m_dPos = DeviceBuffer<Coord>::create(num);

		updateStates();
	}

	template<typename TDataType>
	DensitySimple<TDataType>::~DensitySimple()
	{

	}

	template<typename TDataType>
	bool DensitySimple<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attArr = m_parent->GetAttributeBuffer()->getDataPtr();
		DeviceArray<NeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();
		float dt = m_parent->getDt();

		DeviceArray<Coord>* dPos = m_dPos->getDataPtr();

		dim3 pDims = int(ceil(posArr->Size() / BLOCK_SIZE + 0.5f));

		Real samplingDistance = 0.5f*m_parent->GetSamplingDistance();
		Real smoothingLength = 0.9f*m_parent->GetSamplingDistance();

		int it = 0;
		while (it < 5)
		{
			dPos->Reset();
			DIS_ComputeDelta <Real, Coord> << <pDims, BLOCK_SIZE >> > (*dPos, *posArr, *neighborArr, smoothingLength, samplingDistance);
			DIS_UpdatePosition <Real, Coord> << <pDims, BLOCK_SIZE >> > (*posArr, *velArr, *dPos, *attArr, dt);

			it++;
		}

		return true;
	}

	template<typename TDataType>
	bool DensitySimple<TDataType>::updateStates()
	{
// 		DIS_STATE cm;
// 		cm.samplingDistance = 0.5f*m_parent->GetSamplingDistance();
// 		cm.smoothingLength = 0.9f*m_parent->GetSamplingDistance();
// 		cm.kernSpiky = SpikyKernel();
// 		cudaMemcpyToSymbol(const_dis_state, &cm, sizeof(DIS_STATE));

		return true;
	}

}


