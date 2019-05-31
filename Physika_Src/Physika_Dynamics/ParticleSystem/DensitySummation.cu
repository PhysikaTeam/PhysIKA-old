#include <cuda_runtime.h>
#include "DensitySummation.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Core/Utility.h"
#include "Kernel.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(DensitySummation, TDataType)

	template<typename Real, typename Coord>
	__global__ void K_ComputeDensity(
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		SpikyKernel<Real> kern;
		Real r;
		Real rho_i = Real(0);
		Coord pos_i = posArr[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			r = (pos_i - posArr[j]).norm();
			rho_i += mass*kern.Weight(r, smoothingLength);
		}
		rhoArr[pId] = rho_i;
	}

	template<typename TDataType>
	DensitySummation<TDataType>::DensitySummation()
		: ComputeModule()
		, m_factor(Real(1))
	{
		m_mass.setValue(Real(1));
		m_restDensity.setValue(Real(1000));
		m_smoothingLength.setValue(Real(0.011));

		attachField(&m_mass, "mass", "particle mass", false);
		attachField(&m_restDensity, "rest_density", "Reference density", false);
		attachField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);

		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_density, "density", "Storing the particle densities!", false);
		attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);
	}

	template<typename TDataType>
	void DensitySummation<TDataType>::compute()
	{
		compute(
			m_density.getValue(),
			m_position.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue(),
			m_mass.getValue());
	}


	template<typename TDataType>
	void DensitySummation<TDataType>::compute(DeviceArray<Real>& rho)
	{
		compute(
			rho,
			m_position.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue(),
			m_mass.getValue());
	}

	template<typename TDataType>
	void DensitySummation<TDataType>::compute(
		DeviceArray<Real>& rho, 
		DeviceArray<Coord>& pos,
		NeighborList<int>& neighbors, 
		Real smoothingLength,
		Real mass)
	{
		cuint pDims = cudaGridSize(rho.size(), BLOCK_SIZE);
		K_ComputeDensity <Real, Coord> << <pDims, BLOCK_SIZE >> > (rho, pos, neighbors, smoothingLength, m_factor*mass);
	}

	template<typename TDataType>
	bool DensitySummation<TDataType>::initializeImpl()
	{
		if (!m_position.isEmpty() && m_density.isEmpty())
		{
			m_density.setElementCount(m_position.getElementCount());
		}

		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("DensitySummation's fields are not fully initialized!") << "\n";
			return false;
		}

		compute(
			m_density.getValue(),
			m_position.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue(),
			m_mass.getValue());

		auto rho = m_density.getReference();

		Reduction<Real>* pReduce = Reduction<Real>::Create(rho->size());

		Real maxRho = pReduce->Maximum(rho->getDataPtr(), rho->size());

		m_factor = m_restDensity.getValue() / maxRho;
		
		delete pReduce;

		return true;
	}
}