#include <vector>
#include <string>

#include "Core/Vector/vector_3d.h"
#include "Dynamics/RigidBody/RigidCollisionBody.h"

using namespace std;
namespace PhysIKA {
IMPLEMENT_CLASS_1(RigidCollisionBody, TDataType)
template <typename TDataType>
PhysIKA::RigidCollisionBody<TDataType>::RigidCollisionBody()
{
    meshPtr          = std::make_shared<TriangleMesh<TDataType>>();
    getSurfaceNode() = this->createChild<Node>("Mesh");
    auto triSet      = getSurfaceNode()->template setTopologyModule<TriangleSet<TDataType>>("surface_mesh");
}
template <typename TDataType>
void PhysIKA::RigidCollisionBody<TDataType>::postprocess()
{
    this->setVisible(true);
}
template <typename TDataType>
void PhysIKA::RigidCollisionBody<TDataType>::translate(Coord t)
{
    //gpu
    TypeInfo::CastPointerDown<TriangleSet<TDataType>>(getSurfaceNode()->getTopologyModule())->translate(t);
    //cpu
    auto& points = meshPtr->getTriangleSet()->gethPoints();
    for (int i = 0; i < points.size(); ++i)
    {
        *&points[i] += t;
    }
    //meshPtr->translate(t);
}
template <typename TDataType>
std::shared_ptr<TriangleMesh<TDataType>>& PhysIKA::RigidCollisionBody<TDataType>::getmeshPtr()
{
    return meshPtr;
}
template <typename TDataType>
std::shared_ptr<Node>& PhysIKA::RigidCollisionBody<TDataType>::getSurfaceNode()
{
    return meshPtr->getSurfaceNode();
}

}  // namespace PhysIKA
