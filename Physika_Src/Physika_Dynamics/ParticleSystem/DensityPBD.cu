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
		DeviceArray<Real> massInvArr,
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
				lamda_i += g.dot(g) * massInvArr[j];
			}
		}

		lamda_i += grad_ci.dot(grad_ci) * massInvArr[pId];

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacement(
		DeviceArray<Coord> dPos, 
		DeviceArray<Real> lambdas, 
		DeviceArray<Coord> posArr, 
		DeviceArray<Real> massInvArr,
		NeighborList<int> neighbors, 
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		SpikyKernel<Real> kern;

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 1.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
				Coord dp_ji = - dp_ij * massInvArr[j];
				dp_ij = dp_ij * massInvArr[pId];
				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], dp_ji[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], dp_ji[1]);
				}
				
				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], dp_ji[2]);
				}
			}
		}
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
		, m_maxIteration(5)
	{
		m_restDensity.setValue(Real(1000));
		m_smoothingLength.setValue(Real(0.011));

		initField(&m_restDensity, "rest_density", "Reference density", false);
		initField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);
		initField(&m_position, "position", "Storing the particle positions!", false);
		initField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		initField(&m_density, "density", "Storing the particle densities!", false);
		initField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);
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
		int num = m_position.getElementCount();

		Real dt = getParent()->getDt();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		int it = 0;
		while (it < m_maxIteration)
		{
			m_deltaPos.reset();
			m_densitySum->compute(m_rhoArr);

			K_ComputeLambdas <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_lamda, 
				m_rhoArr, 
				m_position.getValue(), 
				m_massInv.getValue(),
				m_neighborhood.getValue(), 
				m_smoothingLength.getValue());
			K_ComputeDisplacement <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_deltaPos, 
				m_lamda, 
				m_position.getValue(),
				m_massInv.getValue(),
				m_neighborhood.getValue(),
				m_smoothingLength.getValue(), 
				dt);
			K_UpdatePosition <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_position.getValue(),
				m_velocity.getValue(), 
				m_deltaPos, 
				dt);

			it++;
		}

		return true;
	}
}