#include "CollidableTriangleMesh.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Collision/CollisionBVH.h"
namespace PhysIKA {

IMPLEMENT_CLASS_1(CollidatableTriangleMesh, TDataType)

template <typename TDataType>
bool CollidatableTriangleMesh<TDataType>::initializeImpl()
{
}
template <typename TDataType>
CollidatableTriangleMesh<TDataType>::CollidatableTriangleMesh()
    : CollidableObject(CollidableObject::TRIANGLE_TYPE)
{
}
template <typename TDataType>
CollidatableTriangleMesh<TDataType>::~CollidatableTriangleMesh()
{
}
template <typename TDataType>
void CollidatableTriangleMesh<TDataType>::updateCollidableObject()
{
}
template <typename TDataType>
void CollidatableTriangleMesh<TDataType>::updateMechanicalState()
{
}

// add by HNU
std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> CollisionManager::Meshes;  //!< storage for input mesh
}  // namespace PhysIKA