#pragma once
#include "Physika_Core/Cuda_Array/Array3D.h"

namespace Physika
{
	template<typename Coord>
	class UniformGrid3D
	{
	public:
		UniformGrid3D() {};
		~UniformGrid3D() {};

	private:
		DeviceArray3D<Coord> m_coords;
	};
}


