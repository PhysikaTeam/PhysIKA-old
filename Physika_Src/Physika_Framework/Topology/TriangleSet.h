#pragma once
#include "EdgeSet.h"
#include "Framework/ModuleTopology.h"


namespace Physika
{
	template<typename TDataType>
	class TriangleSet : public EdgeSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		TriangleSet();
		~TriangleSet();

		DeviceArray<Triangle>* getTriangles() { return &m_triangls; }
		void setTriangles(std::vector<Triangle>& triangles);

		NeighborList<int>* getTriangleNeighbors() { return &m_triangleNeighbors; }

		void updatePointNeighbors() override;

	protected:
		bool initializeImpl() override;

	protected:
		DeviceArray<Triangle> m_triangls;
		NeighborList<int> m_triangleNeighbors;
	};

#ifdef PRECISION_FLOAT
	template class TriangleSet<DataType3f>;
#else
	template class TriangleSet<DataType3d>;
#endif
}

