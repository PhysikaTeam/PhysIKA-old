#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "SummationDensity.h"

namespace Physika
{
// 	struct SD_STATE
// 	{
// 		float mass;
// 		float smoothingLength;
// 		SpikyKernel kernSpiky;
// 	};

//	__constant__ SD_STATE const_sd_state;

	template<typename Real, typename Coord>
	__global__ void SD_ComputeDensity(
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		DeviceArray<NeighborList> neighbors,
		Real smoothingLength,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		SpikyKernel<Real> kern;
		Real r;
		Real rho_i = 0.0f;
		Coord pos_i = posArr[pId];
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			r = (pos_i - posArr[j]).norm();

			rho_i += mass*kern.Weight(r, smoothingLength);
		}
		rhoArr[pId] = rho_i;
	}

	template<typename TDataType>
	SummationDensity<TDataType>::SummationDensity(ParticleSystem<TDataType>* parent)
		:Module()
		,m_parent(parent)
		,m_factor(1.0f)
	{
		assert(m_parent != NULL);

		setInputSize(1);
		setOutputSize(1);

		updateStates();
	}

	template<typename TDataType>
	bool SummationDensity<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<Real>* rhoArr = m_parent->GetDensityBuffer()->getDataPtr();
		DeviceArray<NeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();
		float dt = m_parent->getDt();

		Real mass = m_factor * m_parent->GetParticleMass();
		Real smoothingLength = m_parent->GetSmoothingLength();

		uint pDims = cudaGridSize(posArr->Size(), BLOCK_SIZE);
		SD_ComputeDensity <Real, Coord> << <pDims, BLOCK_SIZE >> > (*rhoArr, *posArr, *neighborArr, smoothingLength, mass);

		return true;
	}

	template<typename TDataType>
	bool SummationDensity<TDataType>::updateStates()
	{
// 		SD_STATE cm;
// 		cm.mass = m_factor * m_parent->GetParticleMass();
// 		cm.smoothingLength = m_parent->GetSmoothingLength();
// 		cm.kernSpiky = SpikyKernel();
// 		
// 		cudaMemcpyToSymbol(const_sd_state, &cm, sizeof(SD_STATE));

		return true;
	}

}