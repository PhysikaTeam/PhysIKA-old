#pragma once
#include "EdgeSet.h"
#include "Framework/Framework/ModuleTopology.h"

namespace PhysIKA {
template <typename TDataType>
class TriangleSet : public EdgeSet<TDataType>
{
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    TriangleSet();
    ~TriangleSet();
    TriangleSet(TriangleSet<TDataType>& triangleset);
    TriangleSet& operator=(TriangleSet<TDataType>& triangleset);

    DeviceArray<Triangle>* getTriangles()
    {
        return &m_triangls;
    }
    std::vector<Triangle>& getHTriangles()
    {
        return h_triangles;
    }
    void setTriangles(std::vector<Triangle>& triangles);

    NeighborList<int>* getTriangleNeighbors()
    {
        return &m_triangleNeighbors;
    }

    void updatePointNeighbors() override;

    void loadObjFile(std::string filename);

    template <typename TDataType>
    friend class TriangleMesh;

protected:
    bool initializeImpl() override;

protected:
    std::vector<Triangle> h_triangles;
    DeviceArray<Triangle> m_triangls;
    NeighborList<int>     m_triangleNeighbors;
};

#ifdef PRECISION_FLOAT
template class TriangleSet<DataType3f>;
#else
template class TriangleSet<DataType3d>;
#endif
}  // namespace PhysIKA
