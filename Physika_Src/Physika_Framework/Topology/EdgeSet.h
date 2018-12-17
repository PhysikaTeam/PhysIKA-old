#pragma once
#include "PointSet.h"

namespace Physika
{
	template<typename TDataType>
	class EdgeSet : public PointSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		EdgeSet();
		~EdgeSet() override;

		void updatePointNeighbors() override;

		DeviceArray<Edge>* getEdges() {return &m_edges;}
		NeighborList<int>* getEdgeNeighbors() { return &m_edgeNeighbors; }

	protected:
		DeviceArray<Edge> m_edges;
		NeighborList<int> m_edgeNeighbors;
	};

#ifdef PRECISION_FLOAT
	template class EdgeSet<DataType3f>;
#else
	template class EdgeSet<DataType3d>;
#endif
}

