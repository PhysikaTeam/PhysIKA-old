#pragma once
#include "Physika_Framework/Framework/TopologyMapping.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Framework/Topology/PointSet.h"

namespace Physika
{
	template<typename TDataType> class PointSet;

template<typename TDataType>
class PointSetToPointSet : public TopologyMapping
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	PointSetToPointSet();
	~PointSetToPointSet() override;

	void setSearchingRadius(Real r) { m_radius = r; }

	void setFrom(PointSet<TDataType>* from) { m_from = from; }
	void setTo(PointSet<TDataType>* to) { m_to = to; }

	bool apply() override;

	void initialize(DeviceArray<Coord>& from, DeviceArray<Coord>& to);

	void applyTranform(DeviceArray<Coord>& from, DeviceArray<Coord>& to);
private:
	//Searching radius
	Real m_radius;

	NeighborList<int> m_neighborhood;
	
	PointSet<TDataType>* m_from = nullptr;
	PointSet<TDataType>* m_to = nullptr;
};


#ifdef PRECISION_FLOAT
template class PointSetToPointSet<DataType3f>;
#else
template class PointSetToPointSet<DataType3d>;
#endif
}