#pragma once
#include "Framework/Framework/TopologyMapping.h"
#include "Core/Array/Array.h"
#include "Framework/Topology/PointSet.h"

namespace PhysIKA {
template <typename TDataType>
class PointSet;

template <typename TDataType>
class PointSetToPointSet : public TopologyMapping
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    PointSetToPointSet();
    PointSetToPointSet(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);
    ~PointSetToPointSet() override;

    void setSearchingRadius(Real r)
    {
        m_radius = r;
    }

    void setFrom(std::shared_ptr<PointSet<TDataType>> from)
    {
        m_from = from;
    }
    void setTo(std::shared_ptr<PointSet<TDataType>> to)
    {
        m_to = to;
    }

    bool apply() override;

    void match(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);

protected:
    bool initializeImpl() override;

private:
    //Searching radius
    Real m_radius = 0.0125;

    NeighborList<int> m_neighborhood;

    std::shared_ptr<PointSet<TDataType>> m_from = nullptr;
    std::shared_ptr<PointSet<TDataType>> m_to   = nullptr;

    std::shared_ptr<PointSet<TDataType>> m_initFrom = nullptr;
    std::shared_ptr<PointSet<TDataType>> m_initTo   = nullptr;
};

#ifdef PRECISION_FLOAT
template class PointSetToPointSet<DataType3f>;
#else
template class PointSetToPointSet<DataType3d>;
#endif
}  // namespace PhysIKA