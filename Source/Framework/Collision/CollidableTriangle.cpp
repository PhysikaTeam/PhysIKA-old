#include "CollidableTriangle.h"

namespace PhysIKA {

template <typename TDataType>
CollidableTriangle<TDataType>::CollidableTriangle()
    : CollidableObject(CollidableObject::TRIANGLE_TYPE)
{
}
template <typename TDataType>
CollidableTriangle<TDataType>::~CollidableTriangle()
{
}
}  // namespace PhysIKA