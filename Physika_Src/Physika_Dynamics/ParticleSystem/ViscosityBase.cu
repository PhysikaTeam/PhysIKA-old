#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "ViscosityBase.h"
#include "Attribute.h"
#include "Physika_Framework/Topology/INeighbors.h"
#include "Physika_Framework/Framework/Node.h"


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
		DeviceArray<SPHNeighborList> neighbors,
		DeviceArray<Attribute> attArr,
		Real viscosity,
		Real smoothingLength,
		Real samplingDistance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;
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
		if (pId >= velArr.size()) return;

		velArr[pId] = dVel[pId];
	}

	template<typename TDataType>
	ViscosityBase<TDataType>::ViscosityBase()
		:ForceModule()
		, m_oldVel(NULL)
		, m_bufVel(NULL)
	{
		m_viscosity = HostVariable<Real>::createField(this, "viscosity", "Viscosity", Real(0.05));

		initArgument(&m_position, "Position", "CUDA array used to store particles' positions");
		initArgument(&m_velocity, "Velocity", "CUDA array used to store particles' velocities");
		initArgument(&m_density, "Density", "CUDA array used to store particles' densities");
		initArgument(&m_radius, "Radius", "Smoothing length");
		initArgument(&m_neighbors, "Neighbors", "Neighbors");
		initArgument(&m_samplingDistance, "SamplingDistance", "Sampling distance");
		initArgument(&m_attribute, "Attribute", "Particle attribute");
	}

	template<typename TDataType>
	bool ViscosityBase<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_position.getField().getDataPtr();
		DeviceArray<Coord>* velArr = m_velocity.getField().getDataPtr();
		DeviceArray<Attribute>* attArr = m_attribute.getField().getDataPtr();
		DeviceArray<SPHNeighborList>* neighborArr = m_neighbors.getField().getDataPtr();

		int num = posArr->size();
		if (m_oldVel == NULL)
		{
			m_oldVel = DeviceBuffer<Coord>::create(num);
		}
		if (m_bufVel == NULL)
		{
			m_bufVel = DeviceBuffer<Coord>::create(num);
		}

		DeviceArray<Coord>* oldVel = m_oldVel->getDataPtr();
		DeviceArray<Coord>* bufVel = m_bufVel->getDataPtr();
		Real dt = getParent()->getDt();

		cuint pDims = cudaGridSize(posArr->size(), BLOCK_SIZE);

		Real smoothingLength = m_radius.getField().getValue();
		Real viscosity = m_viscosity->getValue();
		Real samplingDistance = m_samplingDistance.getField().getValue();

		Function1Pt::Copy(*oldVel, *velArr);
		for (int t = 0; t < 5; t++)
		{
			Function1Pt::Copy(*bufVel, *velArr);
			VB_ApplyViscosity <Real, Coord> << < pDims, BLOCK_SIZE >> > (*velArr, *oldVel, *posArr, *bufVel, *neighborArr, *attArr, viscosity, smoothingLength, samplingDistance, dt);
		}
		return true;
	}

}