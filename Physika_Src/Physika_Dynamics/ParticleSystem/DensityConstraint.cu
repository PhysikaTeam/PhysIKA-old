#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "DensityConstraint.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(DensityConstraint, TDataType)

	template<typename TDataType>
	DensityConstraint<TDataType>::DensityConstraint()
		: m_parent(NULL)
	{

	}

	template<typename TDataType>
	DensityConstraint<TDataType>::DensityConstraint(ParticleSystem<TDataType>* parent)
		: m_parent(parent)
	{
		assert(m_parent != NULL);

		setInputSize(2);
		setOutputSize(1);

		int num = m_parent->GetParticleNumber();

		updateStates();
	}

	template<typename TDataType>
	bool DensityConstraint<TDataType>::execute()
	{
		return true;
	}

	template<typename TDataType>
	bool DensityConstraint<TDataType>::updateStates()
	{
		return true;
	}
}