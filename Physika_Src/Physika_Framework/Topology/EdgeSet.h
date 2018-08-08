#pragma once
#include "PointSet.h"

namespace Physika
{
	template<typename Coord>
	class EdgeSet : public PointSet<Coord>
	{
	public:
		EdgeSet();
		~EdgeSet();

		void updatePointNeighbors() override;

		DeviceArray<Edge>* getEdges() {return &m_edges;}
		NeighborList<int>* getEdgeNeighbors() { return &m_edgeNeighbors; }

	protected:
		DeviceArray<Edge> m_edges;
		DeviceArray<int> m_edgeNeighbors;
	};

}

