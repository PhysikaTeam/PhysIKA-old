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

	private:
		DeviceArray<Edge> m_edges;
	};
}

