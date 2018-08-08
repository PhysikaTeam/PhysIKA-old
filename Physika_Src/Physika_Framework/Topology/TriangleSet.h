#pragma once
#include "EdgeSet.h"


namespace Physika
{
	template<typename Coord>
	class TriangleSet : public EdgeSet<Coord>
	{
	public:
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

