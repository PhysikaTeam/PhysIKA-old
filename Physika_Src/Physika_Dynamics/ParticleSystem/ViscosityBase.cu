#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "ViscosityBase.h"


namespace Physika
{
// 	struct VB_STATE
// 	{
// 		float mass;
// 		float smoothingLength;
// 		float samplingDistance;
// 		float viscosity;
// 	};
// 
// 	__constant__ VB_STATE const_vb_state;

	template<typename Real>
	__device__ Real VB_VisWeight(const Real r, const Real h)
	{
		Real q = r / h;
		if (q > 1.0f) return 0.0;
		else {
			const Real d = 1.0f - q;
			const Real RR = h*h;
			return 45.0f / (13.0f * (Real)M_PI * RR *h) *d;
		}
	}

	template<typename Real, typename Coord>
	__global__ void VB_ApplyViscosity(
		DeviceArray<Coord> velNew,
		DeviceArray<Coord> velOld,
		DeviceArray<Coord> posArr,
		DeviceArray<Coord> velArr,
		DeviceArray<NeighborList> neighbors,
		DeviceArray<Attribute> attArr,
		Real viscosity,
		Real smoothingLength,
		Real samplingDistance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;
		if (!attArr[pId].IsDynamic()) return;

		Real r;
		Coord dv_i(0);
		Coord pos_i = posArr[pId];
		Coord vel_i = velArr[pId];
		Real totalWeight = 0.0f;
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Real weight = VB_VisWeight(r, smoothingLength);
				totalWeight += weight;
				dv_i += weight * velArr[j];
			}
		}

		Real b = dt*viscosity / samplingDistance;

		b = totalWeight < EPSILON ? 0.0f : b;

		totalWeight = totalWeight < EPSILON ? 1.0f : totalWeight;

		dv_i /= totalWeight;

		velNew[pId] = velOld[pId] / (1.0f + b) + dv_i*b / (1.0f + b);
	}

	template<typename Real, typename Coord>
	__global__ void VB_UpdateVelocity(
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dVel)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.Size()) return;

		velArr[pId] = dVel[pId];
	}

	template<typename TDataType>
	ViscosityBase<TDataType>::ViscosityBase(ParticleSystem<TDataType>* parent)
		:Module()
		, m_parent(parent)
	{
		assert(m_parent != NULL);

		setInputSize(2);
		setOutputSize(1);

		int num = m_parent->GetParticleNumber();

		m_oldVel = DeviceBuffer<Coord>::create(num);
		m_bufVel = DeviceBuffer<Coord>::create(num);

		updateStates();
	}

	template<typename TDataType>
	bool ViscosityBase<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Coord>* velArr = m_parent->GetNewVelocityBuffer()->getDataPtr();
		DeviceArray<Attribute>* attArr = m_parent->GetAttributeBuffer()->getDataPtr();
		DeviceArray<NeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();
		DeviceArray<Coord>* oldVel = m_oldVel->getDataPtr();
		DeviceArray<Coord>* bufVel = m_bufVel->getDataPtr();
		Real dt = m_parent->getDt();

		uint pDims = cudaGridSize(posArr->Size(), BLOCK_SIZE);

		Real mass = m_parent->GetParticleMass();
		Real smoothingLength = m_parent->GetSmoothingLength();
		Real viscosity = m_parent->GetViscosity();
		Real samplingDistance = m_parent->GetSamplingDistance();

		Function1Pt::Copy(*oldVel, *velArr);
		for (int t = 0; t < 5; t++)
		{
			Function1Pt::Copy(*bufVel, *velArr);
			VB_ApplyViscosity <Real, Coord> << < pDims, BLOCK_SIZE >> > (*velArr, *oldVel, *posArr, *bufVel, *neighborArr, *attArr, viscosity, smoothingLength, samplingDistance, dt);
		}

		return true;
	}

	template<typename TDataType>
	bool ViscosityBase<TDataType>::updateStates()
	{
// 		VB_STATE cm;
// 		cm.mass = m_parent->GetParticleMass();
// 		cm.smoothingLength = m_parent->GetSmoothingLength();
// 		cm.viscosity = m_parent->GetViscosity();
// 		cm.samplingDistance = m_parent->GetSamplingDistance();
// 		cudaMemcpyToSymbol(const_vb_state, &cm, sizeof(VB_STATE));

		return true;
	}

}