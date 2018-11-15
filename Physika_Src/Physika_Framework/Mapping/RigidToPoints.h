#pragma once
#include "Physika_Core/Cuda_Array/Array.h"
#include "Mapping/Mapping.h"

namespace Physika
{
template<typename TDataType>
class RigidToPoints : public Mapping
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Rigid Rigid;
	typedef typename TDataType::Matrix Matrix;

	RigidToPoints();
	~RigidToPoints() override;

	void initialize(Rigid& rigid, DeviceArray<Coord>& points);

	void applyTransform(Rigid& rigid, DeviceArray<Coord>& points);
	void applyInverseTransform(Rigid& rigid, DeviceArray<Coord>& points);
private:
	Rigid m_refRigid;
	DeviceArray<Coord> m_refPoints;
};

#ifdef PRECISION_FLOAT
template class RigidToPoints<DataType3f>;
#else
template class RigidToPoints<DataType3d>;
#endif

}