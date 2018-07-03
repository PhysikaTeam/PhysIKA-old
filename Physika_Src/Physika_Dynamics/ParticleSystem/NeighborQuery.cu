#include <cuda_runtime.h>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "NeighborQuery.h"

namespace Physika
{
	template<typename TDataType>
	NeighborQuery<TDataType>::NeighborQuery(ParticleSystem<TDataType>* parent)
		:Module()
		,m_parent(parent)
	{
		assert(parent != NULL);

		setInputSize(1);
		setOutputSize(1);

		Real samplingDistance = m_parent->GetSamplingDistance();
		Coord lowerBound = m_parent->GetLowerBound();
		Coord upperBound = m_parent->GetUpperBound();
		hash.SetSpace(2 * samplingDistance, lowerBound, upperBound);
	}

	template<typename TDataType>
	bool NeighborQuery<TDataType>::execute()
	{
		DeviceArray<Coord>* posArr = m_parent->GetNewPositionBuffer()->getDataPtr();
		DeviceArray<NeighborList>* neighborArr = m_parent->GetNeighborBuffer()->getDataPtr();
		float dt = m_parent->getDt();

		Real smoothingLength = m_parent->GetSmoothingLength();

		hash.QueryNeighborSlow(*posArr, *neighborArr, smoothingLength, NEIGHBOR_SIZE);

		return true;
	}

}