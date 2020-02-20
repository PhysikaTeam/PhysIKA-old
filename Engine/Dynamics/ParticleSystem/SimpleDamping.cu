#include <cuda_runtime.h>
#include "Core/Utility.h"
#include "Framework/Framework/Log.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Framework/Node.h"
#include "SimpleDamping.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(SimpleDamping, TDataType)

	template<typename TDataType>
	SimpleDamping<TDataType>::SimpleDamping()
		: ConstraintModule()
	{
		this->attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		this->attachField(&m_damping, "Damping Coefficient", "Damping Coefficient!", false);

		m_damping.setValue(0.9f);
	}

	template<typename TDataType>
	SimpleDamping<TDataType>::~SimpleDamping()
	{
	}


	template<typename TDataType>
	bool SimpleDamping<TDataType>::initializeImpl()
	{
		if (m_velocity.isEmpty())
		{
			std::cout << "Exception: " << std::string("SimpleDamping's fields are not fully initialized!") << "\n";
			return false;
		}

		return true;
	}

	template <typename Real, typename Coord>
	__global__ void K_DoDamping(
		DeviceArray<Coord> vel,
		Real coefficient)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vel.size()) return;
		
		vel[pId] *= coefficient;
	}

	template<typename TDataType>
	bool SimpleDamping<TDataType>::constrain()
	{
		uint pDims = cudaGridSize(m_velocity.getValue().size(), BLOCK_SIZE);

		K_DoDamping<< < pDims, BLOCK_SIZE >> > (m_velocity.getValue(), m_damping.getValue());

		return true;
	}


	template<typename TDataType>
	void PhysIKA::SimpleDamping<TDataType>::setDampingCofficient(Real c)
	{
		m_damping.setValue(c);
	}

}