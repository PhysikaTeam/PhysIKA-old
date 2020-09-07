#include <cuda_runtime.h>
#include "DensitySummation.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "Kernel.h"

namespace PhysIKA
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
		, m_factor(Real(0.000044))
	{
		m_mass.setValue(Real(1));
		m_restDensity.setValue(Real(1000));

		this->getSmoothingLength()->setValue(Real(0.011));
		this->getSamplingDistance()->setValue(Real(0.005));


		std::function<void()> callback = std::bind(&DensitySummation<TDataType>::calculateScalingFactor, this);

		this->getSmoothingLength()->setCallBackFunc(callback);
		this->getSamplingDistance()->setCallBackFunc(callback);


		calculateScalingFactor();

		attachField(&m_mass, "mass", "particle mass", false);
		attachField(&m_restDensity, "rest_density", "Reference density", false);
		//attachField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);

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
			this->getSmoothingLength()->getValue(),
			m_mass.getValue());
	}


	template<typename TDataType>
	void DensitySummation<TDataType>::compute(DeviceArray<Real>& rho)
	{
		compute(
			rho,
			m_position.getValue(),
			m_neighborhood.getValue(),
			this->getSmoothingLength()->getValue(),
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
		
//		calculateScalingFactor();

		return true;
	}

	template<typename TDataType>
	void DensitySummation<TDataType>::calculateScalingFactor()
	{
		Real d = this->getSamplingDistance()->getValue();
		Real H = this->getSmoothingLength()->getValue();
		Real rho_0 = m_restDensity.getValue();
		Real m = m_mass.getValue();

		SpikyKernel<Real> kern;

		Real rho_e(0);
		int half_res = H / d + 1;
		for (int i = -half_res; i <= half_res; i++)
			for (int j = -half_res; j <= half_res; j++)
				for (int k = -half_res; k <= half_res; k++)
				{
					Real x = i * d;
					Real y = j * d;
					Real z = k * d;
					Real r = sqrt(x * x + y * y + z * z);
					rho_e += m * kern.Weight(r, H);
				}

		m_factor = rho_0 / rho_e;
	}

}