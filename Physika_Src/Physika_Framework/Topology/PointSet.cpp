#include "PointSet.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
	IMPLEMENT_CLASS_1(PointSet, TDataType)

	template<typename TDataType>
	PointSet<TDataType>::PointSet()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	PointSet<TDataType>::~PointSet()
	{
	}

	template<typename TDataType>
	bool PointSet<TDataType>::initializeImpl()
	{
		if (m_coords.size() <= 0)
		{
			std::vector<Coord> positions;
			for (float x = 0.45; x < 0.55; x += 0.005f) {
				for (float y = 0.2; y < 0.3; y += 0.005f) {
					for (float z = 0.45; z < 0.55; z += 0.005f) {
						positions.push_back(Coord(Real(x), Real(y), Real(z)));
					}
				}
			}
			this->setPoints(positions);
		}

		return true;
	}


	template<typename TDataType>
	void PointSet<TDataType>::setPoints(std::vector<Coord>& pos)
	{
		m_coords.resize(pos.size());
		Function1Pt::Copy(m_coords, pos);

		tagAsChanged();
	}

	template<typename TDataType>
	NeighborList<int>* PointSet<TDataType>::getPointNeighbors()
	{
		if (isTopologyChanged())
		{
			updatePointNeighbors();
		}

		return &m_pointNeighbors;
	}

	template<typename TDataType>
	void PointSet<TDataType>::updatePointNeighbors()
	{
		if (m_coords.isEmpty())
			return;

		
	}
}