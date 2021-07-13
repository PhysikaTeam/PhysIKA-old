#pragma once

#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleTopology.h"
#include "Dynamics/RigidBody/HostNeighborList.h"
#include "Framework/Topology/Primitive3D.h"
#include "Framework/Topology/TriangleSet.h"
#include "Core/Matrix/matrix_3x3.h"

#include <vector>

#include <string>

namespace PhysIKA {

class bvh;
template <typename TDataType>
class TriangleMesh
{
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;
    // no use
    TriangleMesh(const char* path, Vector3f offset, Vector3f axis, Real theta);
    TriangleMesh();
    ~TriangleMesh();

    void buildBVH(std::vector<std::shared_ptr<TriangleMesh<TDataType>>>& meshes);

    void loadFromSet(std::shared_ptr<TriangleSet<DataType3f>> set = nullptr);
    void translate(Coord t);
    // no use
    bool readobjfile(const char* path, Vector3f offset, Vector3f axis, Real theta);

    void                updataBxs();
    TAlignedBox3D<Real> bound();
    bvh*                getBVH();

    std::shared_ptr<Node>& getSurfaceNode()
    {
        return m_surfaceNode;
    }
    std::shared_ptr<TriangleSet<TDataType>>& getTriangleSet()
    {
        return triangleSet;
    }
    template <typename TDataType>
    friend class CollidableTriangle;
    friend class CollisionManager;
    template <typename TDataType>
    friend class CollidatableTriangleMesh;
    friend class bvh;
    friend void mesh_id(int id, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& m, int& mid, int& fid);

public:
    std::shared_ptr<Node>                   m_surfaceNode;
    std::shared_ptr<TriangleSet<TDataType>> triangleSet;

    unsigned int         _num_vtx;
    unsigned int         _num_tri;
    TAlignedBox3D<Real>* _bxs;
    TAlignedBox3D<Real>  _bx;
    Real*                _areas;

    // used by time integration
    //Vector3f *_vtxs;
    //Vector3f *_nrms;
    //tri3f *_tris;

    Vector3f* _ivtxs;  // initial positions
    Vector3f* _ovtxs;  // previous positions
    int*      _fflags;
    bvh*      _bvh;

    Vector3f _offset;
    Vector3f _axis;
    Real     _theta;
    Matrix3f _trf;
};

#ifdef PRECISION_FLOAT
template class TriangleMesh<DataType3f>;
#else
template class TriangleMesh<DataType3d>;
#endif
}  // namespace PhysIKA