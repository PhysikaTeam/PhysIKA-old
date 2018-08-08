#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Typedef.h"
#include "DensityConstraint.h"
#include "Physika_Framework/Framework/Log.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(DensityConstraint, TDataType)

	template<typename TDataType>
	DensityConstraint<TDataType>::DensityConstraint()
	{
		initArgument(&m_position, "Position", "CUDA array used to store particles' positions");
		initArgument(&m_velocity, "Velocity", "CUDA array used to store particles' velocities");
		initArgument(&m_density, "Density", "CUDA array used to store particles' densities");
		initArgument(&m_radius, "Radius", "Smoothing length");
		initArgument(&m_neighbors, "Neighbors", "Neighbors");
	}

	template<typename TDataType>
	bool DensityConstraint<TDataType>::updateStates()
	{
		return true;
	}
}