#include <cuda_runtime.h>
//#include "Physika_Core/Utilities/template_functions.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "DensityPBD.h"
#include "Physika_Framework/Framework/Node.h"
#include <string>
#include "Kernel.h"
#include "DensitySummation.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(DensityPBD, TDataType)

	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdas(
		DeviceArray<Real> lambdaArr,
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];

		SpikyKernel<Real> kern;

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);
			}
		}

		lamda_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacement(
		DeviceArray<Coord> dPos, 
		DeviceArray<Real> lambdas, 
		DeviceArray<Coord> posArr, 
		NeighborList<int> neighbors, 
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		SpikyKernel<Real> kern;

		Coord dP_i(0);
		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 1.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
				dP_i += dp_ij;
				
				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], -dp_ij[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], -dp_ij[1]);
				}
				
				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], -dp_ij[2]);
				}
			}
		}

//		dPos[pId] = dP_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dPos, 
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] += dPos[pId];
		velArr[pId] += dPos[pId] / dt;
	}


	template<typename TDataType>
	DensityPBD<TDataType>::DensityPBD()
		: ConstraintModule()
		, m_maxIteration(3)
	{
		m_restDensity.setValue(Real(1000));
		m_smoothingLength.setValue(Real(0.011));

		attachField(&m_restDensity, "rest_density", "Reference density", false);
		attachField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_density, "density", "Storing the particle densities!", false);
		attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);
	}

	template<typename TDataType>
	DensityPBD<TDataType>::~DensityPBD()
	{
		m_rhoArr.release();
		m_lamda.release();
		m_deltaPos.release();
	}

	template<typename TDataType>
	bool DensityPBD<TDataType>::initializeImpl()
	{
		if (!m_position.isEmpty() && m_density.isEmpty())
		{
			m_density.setElementCount(m_position.getElementCount());
		}

		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("DensityPBD's fields are not fully initialized!") << "\n";
			return false;
		}

		m_densitySum = std::make_shared<DensitySummation<TDataType>>();

		m_restDensity.connect(m_densitySum->m_restDensity);
		m_smoothingLength.connect(m_densitySum->m_smoothingLength);
		m_position.connect(m_densitySum->m_position);
		m_density.connect(m_densitySum->m_density);
		m_neighborhood.connect(m_densitySum->m_neighborhood);

		m_densitySum->initialize();


		int num = m_position.getElementCount();

		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);
		if (m_rhoArr.size() != num)
			m_rhoArr.resize(num);

		return true;
	}

	template<typename TDataType>
	bool DensityPBD<TDataType>::constrain()
	{
		Real dt = getParent()->getDt();

		int it = 0;
		while (it < m_maxIteration)
		{
			takeOneIteration(dt);

			it++;
		}

		return true;
	}


	template<typename TDataType>
	void DensityPBD<TDataType>::takeOneIteration(Real dt)
	{
		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_deltaPos.reset();
		m_densitySum->compute();

		K_ComputeLambdas <Real, Coord> << <pDims, BLOCK_SIZE >> > (
			m_lamda,
			m_density.getValue(),
			m_position.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue());
		K_ComputeDisplacement <Real, Coord> << <pDims, BLOCK_SIZE >> > (
			m_deltaPos,
			m_lamda,
			m_position.getValue(),
			m_neighborhood.getValue(),
			m_smoothingLength.getValue(),
			dt);
		K_UpdatePosition <Real, Coord> << <pDims, BLOCK_SIZE >> > (
			m_position.getValue(),
			m_velocity.getValue(),
			m_deltaPos,
			dt);
	}
}