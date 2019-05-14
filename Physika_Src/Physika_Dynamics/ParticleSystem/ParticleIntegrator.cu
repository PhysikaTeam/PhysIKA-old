#include <cuda_runtime.h>
#include "ParticleIntegrator.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Core/Utilities/cuda_utilities.h"

namespace Physika
{
	template<typename TDataType>
	ParticleIntegrator<TDataType>::ParticleIntegrator()
		:NumericalIntegrator()
	{
		initField(&m_position, "position", "Storing the particle positions!", false);
		initField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		initField(&m_forceDensity, "force", "Particle forces", false);
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::begin()
	{
		Function1Pt::copy(m_prePosition, m_position.getValue());
		Function1Pt::copy(m_preVelocity, m_velocity.getValue());
		
		m_forceDensity.getReference()->reset();
	}

	template<typename TDataType>
	void ParticleIntegrator<TDataType>::end()
	{

	}

	template<typename TDataType>
	bool ParticleIntegrator<TDataType>::initializeImpl()
	{
		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("DensitySummation's fields are not fully initialized!") << "\n";
			return false;
		}

		int num = m_position.getElementCount();

		m_prePosition.resize(num);
		m_preVelocity.resize(num);

		return true;
	}

	template<typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DeviceArray<Coord> vel,
		DeviceArray<Coord> forceDensity,
		Real gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forceDensity.size()) return;

		vel[pId] += dt * (forceDensity[pId] + Coord(0, gravity, 0));
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
		Real dt = getParent()->getDt();
		Real gravity = getParent()->getGravity();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (
			m_velocity.getValue(), 
			m_forceDensity.getValue(),
			gravity,
			dt);

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
		Real dt = getParent()->getDt();
		cuint pDims = cudaGridSize(m_position.getReference()->size(), BLOCK_SIZE);

		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			m_position.getValue(), 
			m_velocity.getValue(), 
			dt);

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