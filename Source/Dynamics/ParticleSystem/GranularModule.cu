#include "GranularModule.h"
#include "SummationDensity.h"

namespace PhysIKA
{
	template<typename TDataType>
	GranularModule<TDataType>::GranularModule()
		: ElastoplasticityModule<TDataType>()
	{
	}


	template<typename TDataType>
	bool GranularModule<TDataType>::initializeImpl()
	{
		m_densitySum = std::make_shared<SummationDensity<TDataType>>();

		this->inHorizon()->connect(m_densitySum->varSmoothingLength());
		this->inPosition()->connect(m_densitySum->inPosition());
		this->inNeighborhood()->connect(m_densitySum->inNeighborIndex());

		m_densitySum->initialize();

		return ElastoplasticityModule<TDataType>::initializeImpl();
	}


	__device__ Real Hardening(Real rho, Real restRho)
	{
		if (rho >= restRho)
		{
			float ratio = rho / restRho;
			//ratio = ratio > 1.1f ? 1.1f : ratio;
			return pow(Real(M_E), Real(ratio - 1.0f));
		}
		else
		{
			return Real(0);
		};
	}

	template <typename Real>
	__global__ void PM_ComputeStiffness(
		DeviceArray<Real> stiffiness,
		DeviceArray<Real> density)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= stiffiness.size()) return;

		stiffiness[i] = Hardening(density[i], Real(1000));
	}

	template<typename TDataType>
	void GranularModule<TDataType>::computeMaterialStiffness()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_densitySum->compute();

		PM_ComputeStiffness << <pDims, BLOCK_SIZE >> > (
			this->m_bulkCoefs,
			m_densitySum->outDensity()->getValue());
		cuSynchronize();
	}
}