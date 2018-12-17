#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/Node.h"

namespace Physika
{
	template<typename TDataType>
	ParticleIntegrator<TDataType>::ParticleIntegrator()
		:NumericalIntegrator()
	{
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::begin()
	{
		auto mstate = getParent()->getMechanicalState();

		auto forceFd = mstate->getField<DeviceArrayField<Coord>>(m_forceID);

		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);
		auto velPreFd = mstate->getField<DeviceArrayField<Coord>>(m_velPreID);

		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto posPreFd = mstate->getField<DeviceArrayField<Coord>>(m_posPreID);

		Function1Pt::copy(velPreFd->getValue(), velFd->getValue());
		Function1Pt::copy(posPreFd->getValue(), posFd->getValue());
		forceFd->reset();
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::end()
	{

	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> force,
		Real mass,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= force.size()) return;

		vel[pId] += dt * force[pId] / mass;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> force,
		DeviceArray<Real> mass,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= force.size()) return;

		vel[pId] += dt * force[pId] / mass[pId];
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updateVelocity()
	{
		auto mstate = getParent()->getMechanicalState();

		auto forceFd = mstate->getField<DeviceArrayField<Coord>>(m_forceID);
		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);

		Real dt = getParent()->getDt();
		cuint pDims = cudaGridSize(velFd->size(), BLOCK_SIZE);

		auto fd = mstate->getField(m_massID);
		if (fd->size() <= 1)
		{
			auto massFd = mstate->getField<HostVarField<Real>>(m_massID);
			K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (velFd->getValue(), forceFd->getValue(), massFd->getValue(), dt);
		}
		else
		{
			auto massFd = mstate->getField<DeviceArrayField<Real>>(m_massID);
			K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (velFd->getValue(), forceFd->getValue(), massFd->getValue(), dt);
		}
		
		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> vel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += dt * vel[pId];
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::updatePosition()
	{
		auto mstate = getParent()->getMechanicalState();

		auto posFd = mstate->getField<DeviceArrayField<Coord>>(m_posID);
		auto velFd = mstate->getField<DeviceArrayField<Coord>>(m_velID);

		Real dt = getParent()->getDt();
		cuint pDims = cudaGridSize(posFd->size(), BLOCK_SIZE);

		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (posFd->getValue(), velFd->getValue(), dt);

		return true;
	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::integrate()
	{
		updateVelocity();
		updatePosition();

		return true;
	}
}