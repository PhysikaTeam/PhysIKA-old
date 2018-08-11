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
	bool PointSet<Coord>::initializeImpl()
	{
		std::vector<Coord> positions;
		for (float x = 0.4; x < 0.6; x += 0.005f) {
			for (float y = 0.1; y < 0.2; y += 0.005f) {
				for (float z = 0.4; z < 0.6; z += 0.005f) {
					positions.push_back(Coord(Real(x), Real(y), Real(z)));
				}
			}
		}
		this->setPoints(positions);

		return true;
	}


	template<typename Coord>
	void PointSet<Coord>::setPoints(std::vector<Coord>& pos)
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