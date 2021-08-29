#pragma once
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Topology/NeighborList.h"
#include "Core/Vector.h"

namespace PhysIKA {
template <typename TDataType>
class HeightField : public TopologyModule
{
    DECLARE_CLASS_1(PointSet, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    HeightField();
    ~HeightField() override;

    void copyFrom(HeightField<TDataType>& pointSet);

    void scale(Real s);
    void scale(Coord s);
    void translate(Coord t);

    void setSpace(Real dx, Real dz);

    Real getDx()
    {
        return m_dx;
    }
    Real getDz()
    {
        return m_dz;
    }

    Coord getOrigin()
    {
        return origin;
    }
    void setOrigin(Coord o)
    {
        origin = o;
    }

    DeviceArray2D<Real>& getHeights()
    {
        return m_height;
    }
    DeviceArray2D<Real>& getTerrain()
    {
        return m_terrain;
    }

protected:
    //         DeviceArray<Coord> m_coords;
    //         DeviceArray<Coord> m_normals;
    //         NeighborList<int> m_pointNeighbors;

    Coord origin;

    Real m_dx;
    Real m_dz;

    DeviceArray2D<Real> m_height;
    DeviceArray2D<Real> m_terrain;
};

#ifdef PRECISION_FLOAT
template class HeightField<DataType3f>;
#else
template class HeightField<DataType3d>;
#endif
}  // namespace PhysIKA
