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

	private:
		DeviceArray<Triangle> m_triangls;
	};

}

