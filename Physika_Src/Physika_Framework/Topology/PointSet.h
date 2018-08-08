#pragma once
#include "Framework/ModuleTopology.h"
#include "Topology/NeighborList.h"
#include "Physika_Core/Vectors/vector.h"


namespace Physika
{
	template<typename Coord>
	class PointSet : public TopologyModule
	{
	public:
		PointSet();
		~PointSet();

		bool initialize() override;
		virtual void initialize(std::vector<Coord>& pos);

		DeviceArray<Coord>* getPoints() { return &m_coords; }
		int getPointSize() { return m_coords.size(); };

		NeighborList<int>* getPointNeighbors();
		virtual void updatePointNeighbors();

	protected:
		DeviceArray<Coord> m_coords;
		NeighborList<int> m_pointNeighbors;
	};


	template class PointSet<Vector3f>;
	template class PointSet<Vector3d>;
}

