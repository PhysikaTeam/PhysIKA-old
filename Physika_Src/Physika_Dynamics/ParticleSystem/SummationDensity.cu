#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "SummationDensity.h"
#include "Attribute.h"

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
		DeviceArray<SPHNeighborList> neighbors,
		Real smoothingLength,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

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
	SummationDensity<TDataType>::SummationDensity()
		:Module()
		, m_factor(1.0f)
	{
		initArgument(&m_position, "Position", "CUDA array used to store particles' positions");
		initArgument(&m_density, "Density", "CUDA array used to store particles' densities");
		initArgument(&m_radius, "Radius", "Smoothing length");
		initArgument(&m_neighbors, "Neighbors", "Neighbors");
		initArgument(&m_mass, "Mass", "Particle mass");
	}

	template<typename TDataType>
	bool SummationDensity<TDataType>::initializeImpl()
	{
		return true;
	}

	template<typename TDataType>
	bool Physika::SummationDensity<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_position.getField().getDataPtr();
		DeviceArray<Real>* rhoArr = m_density.getField().getDataPtr();
		DeviceArray<SPHNeighborList>* neighborArr = m_neighbors.getField().getDataPtr();

		Real mass = m_mass.getField().getValue();
		Real smoothingLength = m_radius.getField().getValue();

		cuint pDims = cudaGridSize(posArr->size(), BLOCK_SIZE);
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