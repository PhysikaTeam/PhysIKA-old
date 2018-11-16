#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Framework/Framework/CollidableObject.h"

namespace Physika
{
	template<typename TDataType>
	class CollidableCube : public CollidableObject
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CollidableCube();
		virtual ~CollidableCube();

	private:
		Coord m_length;
		Coord m_center;
	};



}
