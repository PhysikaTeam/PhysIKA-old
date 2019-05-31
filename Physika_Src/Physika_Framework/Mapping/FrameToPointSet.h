#pragma once
#include "Physika_Core/Array/Array.h"
#include "Physika_Framework/Framework/TopologyMapping.h"

namespace Physika
{
	template<typename TDataType> class Frame;
	template<typename TDataType> class PointSet;

template<typename TDataType>
class FrameToPointSet : public TopologyMapping
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Rigid Rigid;
	typedef typename TDataType::Matrix Matrix;

	FrameToPointSet();
	FrameToPointSet(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);
	~FrameToPointSet() override;

	void initialize(Rigid& rigid, DeviceArray<Coord>& points);

	void applyTransform(Rigid& rigid, DeviceArray<Coord>& points);

	bool apply() override;

protected:
	bool initializeImpl() override;

private:
	void match(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);

	std::shared_ptr<Frame<TDataType>> m_from = nullptr;
	std::shared_ptr<PointSet<TDataType>> m_to = nullptr;

	Rigid m_refRigid;
	DeviceArray<Coord> m_refPoints;

	std::shared_ptr<Frame<TDataType>> m_initFrom;
	std::shared_ptr<PointSet<TDataType>> m_initTo;
};

#ifdef PRECISION_FLOAT
template class FrameToPointSet<DataType3f>;
#else
template class FrameToPointSet<DataType3d>;
#endif

}