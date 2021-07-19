#pragma once
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
namespace PhysIKA {
template <typename TDataType>
class RigidCollisionBody : public RigidBody<TDataType>
{
    DECLARE_CLASS_1(RigidCollisionBody, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;
    RigidCollisionBody();
    virtual ~RigidCollisionBody(){};

    //don't move to header file
    void loadSurface(std::string filename)
    {
        TypeInfo::CastPointerDown<TriangleSet<TDataType>>(getSurfaceNode()->getTopologyModule())->loadObjFile(filename);
        meshPtr->loadFromSet();
        postprocess();
    }

    // friend void                               createScene();
    std::shared_ptr<TriangleMesh<TDataType>>& getmeshPtr();
    std::shared_ptr<Node>&                    getSurfaceNode();
    void                                      postprocess();
    void                                      translate(Coord t);

private:
    std::shared_ptr<TriangleMesh<TDataType>> meshPtr;
};
#ifdef PRECISION_FLOAT
template class RigidCollisionBody<DataType3f>;
#else
template class RigidCollisionBody<DataType3d>;
#endif
}  // namespace PhysIKA