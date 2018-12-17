#pragma once
#include "PointsToPoints.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
	template<typename TDataType>
	PointsToPoints<TDataType>::PointsToPoints()
		: Mapping()
	{

	}

	template<typename TDataType>
	PointsToPoints<TDataType>::~PointsToPoints()
	{

	}


	template<typename TDataType>
	void PointsToPoints<TDataType>::initialize(DeviceArray<Coord>& from, DeviceArray<Coord>& to)
	{
		if (from.size() == to.size())
		{
			return;
		}

	}


	template<typename TDataType>
	void PointsToPoints<TDataType>::applyTranform(DeviceArray<Coord>& from, DeviceArray<Coord>& to)
	{
		if (from.size() == to.size())
		{
			Function1Pt::copy(to, from);
		}
		else
		{

		}

	}
}