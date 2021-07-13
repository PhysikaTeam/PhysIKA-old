#pragma once
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Topology/NeighborList.h"
#include "Core/Vector.h"

namespace PhysIKA {
template <typename TDataType>
class PointSet : public TopologyModule
{
    DECLARE_CLASS_1(PointSet, TDataType)
public:
    typedef typename TDataType::Real   Real;
    typedef typename TDataType::Coord  Coord;
    typedef typename TDataType::Matrix Matrix;

    PointSet();
    ~PointSet() override;

    PointSet(PointSet& pointset);
    PointSet& operator=(PointSet& pointset);

    void copyFrom(PointSet<TDataType>& pointSet);

    void setPoints(std::vector<Coord>& pos);
    void setNormals(std::vector<Coord>& normals);
    void setNeighbors(int maxNum, std::vector<int>& elements, std::vector<int>& index);
    void setSize(int size);

    DeviceArray<Coord>& getPoints()
    {
        return m_coords;
    }
    DeviceArray<Coord>& getNormals()
    {
        return m_normals;
    }

    std::vector<Coord>& gethPoints()
    {
        return h_coords;
    }
    std::vector<Coord>& gethNormals()
    {
        return h_normals;
    }

    const int getPointSize()
    {
        return m_coords.size();
    };

    NeighborList<int>* getPointNeighbors();
    virtual void       updatePointNeighbors();

    void scale(Real s);
    void scale(Coord s);
    void translate(Coord t);
    void rotate(Matrix m);

    void loadObjFile(std::string filename);

    Real computeBoundingRadius();

protected:
    bool initializeImpl() override;

    std::vector<Coord> h_coords;
    std::vector<Coord> h_normals;
    DeviceArray<Coord> m_coords;
    DeviceArray<Coord> m_normals;
    NeighborList<int>  m_pointNeighbors;
};

#ifdef PRECISION_FLOAT
template class PointSet<DataType3f>;
#else
template class PointSet<DataType3d>;
#endif
}  // namespace PhysIKA
