#pragma once
#include <vector_types.h>

template<class TReal, class TCoord>
class DataTypes
{
public:
	typedef TReal Real;
	typedef TCoord Coord;

	static const char* name();
};

/// 1f DOF, single precision
typedef DataTypes<float, float> DataType1f;
template<> inline const char* DataType1f::name() { return "DataType1f"; }

/// 2f DOF, single precision
typedef DataTypes<float, float2> DataType2f;
template<> inline const char* DataType2f::name() { return "DataType2f"; }

/// 3f DOF, single precision
typedef DataTypes<float, float3> DataType3f;
template<> inline const char* DataType3f::name() { return "DataType3f"; }

/// 3f DOF, single precision
typedef DataTypes<float, float4> DataType4f;
template<> inline const char* DataType4f::name() { return "DataType4f"; }

/// 1d DOF, double precision
typedef DataTypes<double, double> DataType1d;
template<> inline const char* DataType1d::name() { return "DataType1d"; }

/// 2d DOF, double precision
typedef DataTypes<double, double2> DataType2d;
template<> inline const char* DataType2d::name() { return "DataType2d"; }

/// 3d DOF, double precision
typedef DataTypes<double, double3> DataType3d;
template<> inline const char* DataType3d::name() { return "DataType3d"; }

/// 4d DOF, double precision
typedef DataTypes<double, double4> DataType4d;
template<> inline const char* DataType4d::name() { return "DataType4d"; }

