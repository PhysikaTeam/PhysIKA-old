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

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = (pos_i - posArr[j])*0.1*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
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
		, m_massID(MechanicalState::mass())
		, m_posID(MechanicalState::position())
		, m_velID(MechanicalState::velocity())
		, m_neighborhoodID(MechanicalState::particle_neighbors())
		, m_smoothingLength(0.0125)
		, m_maxIteration(5)
	{
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
		m_densitySum = getParent()->getModule<DensitySummation<TDataType>>();
		if (m_densitySum == nullptr)
		{
			auto summation = std::make_shared<DensitySummation<TDataType>>();
			summation->setSmoothingLength(m_smoothingLength);

			getParent()->addModule(summation);
		}
		return true;
	}

	template<typename TDataType>
	bool DensityPBD<TDataType>::constrain()
	{
		auto mstate = getParent()->getMechanicalState();
		if (!mstate)
		{
			std::cout << "Cannot find a parent node for DensityPBD!" << std::endl;
			return false;
		}

		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);
		auto neighborFd = mstate->getField<NeighborField<int>>(m_neighborhoodID);

		if (posFd == nullptr || velFd == nullptr || neighborFd == nullptr)
		{
			std::cout << "Incomplete inputs for DensityPBD!" << std::endl;
			return false;
		}

		int num = posFd->size();

		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);
		if (m_rhoArr.size() != num)
			m_rhoArr.resize(num);

		Real dt = getParent()->getDt();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		int it = 0;
		while (it < m_maxIteration)
		{
			m_deltaPos.reset();
			m_densitySum->compute(m_rhoArr);

			K_ComputeLambdas <Real, Coord> << <pDims, BLOCK_SIZE >> > (m_lamda, m_rhoArr, posFd->getValue(), neighborFd->getValue(), m_smoothingLength);
			K_ComputeDisplacement <Real, Coord> << <pDims, BLOCK_SIZE >> > (m_deltaPos, m_lamda, posFd->getValue(), neighborFd->getValue(), m_smoothingLength, dt);
			K_UpdatePosition <Real, Coord> << <pDims, BLOCK_SIZE >> > (posFd->getValue(), velFd->getValue(), m_deltaPos, dt);

			it++;
		}

		return true;
	}
}