#pragma once
#include <iostream>
#include "rigid_base.h"
#include "../Vector.h"
#include "../Matrix.h"

namespace Physika {
	template <typename Scalar>
	class Rigid<Scalar, 2>
	{
	public:
		typedef typename Vector<Scalar, 2> TranslationDOF;
		typedef typename Scalar RotationDOF;

		COMM_FUNC Rigid()
			: m_p(0)
			, m_angle(0)
		{};

		COMM_FUNC Rigid(Vector<Scalar, 2> p, Scalar angle)
			: m_p(p)
			, m_angle(angle)
		{};

		COMM_FUNC ~Rigid() {};

		COMM_FUNC Scalar getOrientation() { return m_angle; }
		COMM_FUNC Vector<Scalar, 2> getCenter() { return m_p; }
		COMM_FUNC SquareMatrix<Scalar, 2> getRotationMatrix() { return SquareMatrix<Scalar, 2>(cos(m_angle), -sin(m_angle),
																		  sin(m_angle),  cos(m_angle)); }
	private:
		Vector<Scalar, 2> m_p;
		Scalar m_angle;
	};

	template class Rigid<float, 2>;
	template class Rigid<double, 2>;
	//convenient typedefs 
	typedef Rigid<float, 2> Rigid2f;
	typedef Rigid<double, 2> Rigid2d;

}  //end of namespace Physika
