#pragma once
#include "PointSetToPointSet.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
	template<typename TDataType>
	PointSetToPointSet<TDataType>::PointSetToPointSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	PointSetToPointSet<TDataType>::~PointSetToPointSet()
	{

	}


	template<typename TDataType>
	void PointSetToPointSet<TDataType>::initialize(DeviceArray<Coord>& from, DeviceArray<Coord>& to)
	{
		if (from.size() == to.size())
		{
			return;
		}

	}


	template<typename TDataType>
	void PointSetToPointSet<TDataType>::applyTranform(DeviceArray<Coord>& from, DeviceArray<Coord>& to)
	{
		if (from.size() == to.size())
		{
			Function1Pt::copy(to, from);
		}
		else
		{

		}

	}

	template<typename TDataType>
	bool PointSetToPointSet<TDataType>::apply()
	{
		return true;
	}
}