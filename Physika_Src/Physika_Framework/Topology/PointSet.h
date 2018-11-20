#pragma once
#include "Physika_Framework/Framework/ModuleTopology.h"
#include "Physika_Framework/Topology/NeighborList.h"
#include "Physika_Core/Vectors/vector.h"


namespace Physika
{
	template<typename TDataType>
	class PointSet : public TopologyModule
	{
		DECLARE_CLASS_1(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSet();
		~PointSet();

		virtual void setPoints(std::vector<Coord>& pos);

		DeviceArray<Coord>* getPoints() { return &m_coords; }
		int getPointSize() { return m_coords.size(); };

		NeighborList<int>* getPointNeighbors();
		virtual void updatePointNeighbors();

	protected:
		bool initializeImpl() override;

		DeviceArray<Coord> m_coords;
		NeighborList<int> m_pointNeighbors;
	};


#ifdef PRECISION_FLOAT
	template class PointSet<DataType3f>;
#else
	template class PointSet<DataType3d>;
#endif
}

