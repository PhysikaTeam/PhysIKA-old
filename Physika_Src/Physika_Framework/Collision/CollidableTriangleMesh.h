#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Framework/CollidableObject.h"

namespace Physika
{
	template<typename TDataType>
	class CollidableSpheres : public CollidableObject
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CollidableSpheres();
		virtual ~CollidableSpheres();

		void setRadius(Real radius) { m_radius = radius; }
		void setCenters(DeviceArray<Coord>& centers);

	private:
		Real m_radius;
		DeviceArray<Coord> m_centers;
	};



}
