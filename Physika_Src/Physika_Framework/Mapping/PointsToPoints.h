#pragma once
#include "Mapping/Mapping.h"
#include "Physika_Core/Cuda_Array/Array.h"

namespace Physika
{

template<typename TDataType>
class PointsToPoints : public Mapping
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	PointsToPoints();
	~PointsToPoints() override;

	void initialize(DeviceArray<Coord>& from, DeviceArray<Coord>& to);

	void applyTranform(DeviceArray<Coord>& from, DeviceArray<Coord>& to);
private:

};

#ifdef PRECISION_FLOAT
template class PointsToPoints<DataType3f>;
#else
template class PointsToPoints<DataType3d>;
#endif
}