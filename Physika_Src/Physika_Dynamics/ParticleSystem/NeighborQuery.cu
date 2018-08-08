#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Framework/Framework/Node.h"
#include "NeighborQuery.h"

namespace Physika
{
	template<typename TDataType>
	Physika::NeighborQuery<TDataType>::NeighborQuery()
		:Module()
	{
		Real samplingDistance = 0.0125;
		Coord lowerBound(0);
		Coord upperBound(1.0);
		hash.SetSpace(2 * samplingDistance, lowerBound, upperBound);

		initArgument(&m_position, "Position", "CUDA array used to store particles' positions");
		initArgument(&m_neighbors, "Neighbors", "Neighbors");
		initArgument(&m_samplingDistance, "SamplingDistance", "Sampling Distance");
		initArgument(&m_smoothingLength, "SmoothingLength", "Smoothing Length");
	}

	template<typename TDataType>
	bool Physika::NeighborQuery<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_position.getField().getDataPtr();
		DeviceArray<SPHNeighborList>* neighborArr = m_neighbors.getField().getDataPtr();

		Real smoothingLength = m_smoothingLength.getField().getValue();

		hash.QueryNeighborSlow(*posArr, *neighborArr, smoothingLength, NEIGHBOR_SIZE);

		return true;
	}


}