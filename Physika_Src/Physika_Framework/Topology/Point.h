#pragma once
#include "Physika_Framework/Framework/ModuleTopology.h"
#include "Physika_Core/Vectors/vector.h"


namespace Physika
{
	template<typename TDataType>
	class Point : public TopologyModule
	{
		DECLARE_CLASS_1(Point, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		Point() {};
		~Point() {};

		void setPoint(Coord pos) { m_coord = pos; }

		void setOrientation(Matrix mat) { m_rotation = mat; }

	protected:
		Coord m_coord;
		Matrix m_rotation;
	};

	template class Point<DataType3f>;
	template class Point<DataType3d>;
}

