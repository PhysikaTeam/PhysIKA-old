#include "PointSet.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
	template<typename Coord>
	PointSet<Coord>::PointSet()
		: TopologyModule()
	{
	}

	template<typename Coord>
	PointSet<Coord>::~PointSet()
	{
	}

	template<typename Coord>
	bool PointSet<Coord>::initialize()
	{
		return true;
	}


	template<typename Coord>
	void PointSet<Coord>::initialize(std::vector<Coord>& pos)
	{
		m_coords.resize(pos.size());
		Function1Pt::Copy(m_coords, pos);

		tagAsChanged();
	}

	template<typename Coord>
	NeighborList<int>* PointSet<Coord>::getPointNeighbors()
	{
		if (isTopologyChanged())
		{
			updatePointNeighbors();
		}

		return &m_pointNeighbors;
	}

	template<typename Coord>
	void PointSet<Coord>::updatePointNeighbors()
	{
		if (m_coords.isEmpty())
			return;

		
	}
}