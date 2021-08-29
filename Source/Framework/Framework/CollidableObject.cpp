#include "CollidableObject.h"

namespace PhysIKA {
CollidableObject::CollidableObject(CType ctype)
{
    m_type = ctype;
}

CollidableObject::~CollidableObject()
{
}
}  // namespace PhysIKA