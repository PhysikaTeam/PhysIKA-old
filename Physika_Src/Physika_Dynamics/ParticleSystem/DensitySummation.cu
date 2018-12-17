#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "DensitySummation.h"
#include "Physika_Framework/Framework/MechanicalState.h"
#include "Physika_Framework/Framework/Node.h"
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
		, m_massID(MechanicalState::mass())
		, m_posID(MechanicalState::position())
		, m_rhoID(MechanicalState::density())
		, m_neighborID(MechanicalState::particle_neighbors())
		, m_smoothingLength(Real(0.0125))
		, m_factor(Real(1))
	{
	}

	template<typename TDataType>
	void DensitySummation<TDataType>::compute()
	{
		auto mstate = getParent()->getMechanicalState();
		if (!mstate)
		{
			std::cout << "Cannot find a parent node for SummationDensity!" << std::endl;
		}

		auto massFd = mstate->getField<HostVarField<Real>>(m_massID);
		auto rhoFd = mstate->getField<DeviceArrayField<Real>>(m_rhoID);
		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto neighborFd = mstate->getField<NeighborField<int>>(m_neighborID);

		if (rhoFd == nullptr || posFd == nullptr || neighborFd == nullptr)
		{
			std::cout << "Incomplete inputs for SummationDensity!" << std::endl;
			return;
		}

		compute(rhoFd->getValue(), posFd->getValue(), neighborFd->getValue(), m_smoothingLength, massFd->getValue());
	}


	template<typename TDataType>
	void DensitySummation<TDataType>::compute(DeviceArray<Real>& rho)
	{
		auto mstate = getParent()->getMechanicalState();
		if (!mstate)
		{
			std::cout << "Cannot find a parent node for SummationDensity!" << std::endl;
		}

		auto massFd = mstate->getField<HostVarField<Real>>(m_massID);
		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto neighborFd = mstate->getField<NeighborField<int>>(m_neighborID);

		if (massFd == nullptr || posFd == nullptr || neighborFd == nullptr)
		{
			std::cout << "Incomplete inputs for SummationDensity!" << std::endl;
			return;
		}

		if (rho.size() != posFd->size())
		{
			std::cout << "The size of density array does not match the size of the position array!" << std::endl;
		}
		
		compute(rho, posFd->getValue(), neighborFd->getValue(), m_smoothingLength, massFd->getValue());
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
		K_ComputeDensity <Real, Coord> << <pDims, BLOCK_SIZE >> > (rho, pos, neighbors, smoothingLength, mass);
	}

}