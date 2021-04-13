#pragma once
#define GLM_FORCE_PURE
#include <vector_types.h>
#include "Core/Vector.h"
#include "Core/Matrix.h"
#include "Core/Rigid/rigid.h"

namespace PhysIKA
{
	template<class TReal, class TCoord, class TMatrix, class TRigid>
	class DataTypes
	{
	public:
		typedef TReal Real;
		typedef TCoord Coord;
		typedef TMatrix Matrix;
		typedef TRigid Rigid;

		static const char* getName();
	};

	/// 1f DOF, single precision
	typedef DataTypes<float, float, float, Rigid<float, 1>> DataType1f;
	template<> inline const char* DataType1f::getName() { return "DataType1f"; }

	/// 2f DOF, single precision
	typedef DataTypes<float, Vector2f, Matrix2f, Rigid2f> DataType2f;
	template<> inline const char* DataType2f::getName() { return "DataType2f"; }

	/// 3f DOF, single precision
	typedef DataTypes<float, Vector3f, Matrix3f, Rigid3f> DataType3f;
	template<> inline const char* DataType3f::getName() { return "DataType3f"; }

	/// 1d DOF, double precision
	typedef DataTypes<double, float, float, Rigid<double, 1>> DataType1d;
	template<> inline const char* DataType1d::getName() { return "DataType1d"; }

	/// 2d DOF, double precision
	typedef DataTypes<double, Vector2d, Matrix2d, Rigid2d> DataType2d;
	template<> inline const char* DataType2d::getName() { return "DataType2d"; }

	/// 3d DOF, double precision
	typedef DataTypes<double, Vector3d, Matrix3d, Rigid3d> DataType3d;
	template<> inline const char* DataType3d::getName() { return "DataType3d"; }
}


