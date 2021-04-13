#pragma once
#include "PointSet.h"
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA
{
	template<typename TDataType>
	class EdgeSet : public PointSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;

		EdgeSet();
		~EdgeSet() override;

		void updatePointNeighbors() override;

		void loadSmeshFile(std::string filename);

		DeviceArray<Edge>* getEdges() {return &m_edges;}
		NeighborList<int>& getEdgeNeighbors() { return m_edgeNeighbors.getValue(); }

		NeighborField<int> m_edgeNeighbors;

	protected:
		DeviceArray<Edge> m_edges;
	};

}

