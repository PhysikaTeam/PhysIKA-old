#pragma once
#include "Framework/ModuleTopology.h"


namespace Physika
{
	template<typename Coord>
	class PointSet : public TopologyModule
	{
	public:
		PointSet();
		~PointSet();

	private:
		DeviceArray<Coord> m_coords;
		DeviceArray<Point> m_pids;
	};

}

