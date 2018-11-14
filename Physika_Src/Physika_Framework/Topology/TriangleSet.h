#pragma once
#include "EdgeSet.h"


namespace Physika
{
	template<typename TDataType>
	class TriangleSet : public EdgeSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		TriangleSet();
		~TriangleSet();

		DeviceArray<Triangle>* getTriangles() { return &m_triangls; }

		NeighborList<int>* getTriangleNeighbors() { return &m_triangleNeighbors; }

		void updatePointNeighbors() override;

	protected:
		DeviceArray<Triangle> m_triangls;
		DeviceArray<int> m_triangleNeighbors;
	};

}

