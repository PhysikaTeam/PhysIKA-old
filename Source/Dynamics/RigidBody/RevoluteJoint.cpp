#include "RevoluteJoint.h"
#include "Dynamics/RigidBody/RigidUtil.h"

namespace PhysIKA {
//IMPLEMENT_CLASS_1(RevoluteJoint, TDataType)

PhysIKA::RevoluteJoint::RevoluteJoint(std::string name)
    : Joint(name)
{
}

RevoluteJoint::RevoluteJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void RevoluteJoint::setJointInfo(const Vector3f& axis, const Vector3f& r)
{
    //Transform3d<float> trans(r, Quaternion<float>());
    Vector3f unit_axis(axis[0], axis[1], axis[2]);
    unit_axis.normalize();

    Vector3f v = unit_axis.cross(r);

    this->m_S(0, 0) = unit_axis[0];
    this->m_S(1, 0) = unit_axis[1];
    this->m_S(2, 0) = unit_axis[2];
    this->m_S(3, 0) = -v[0];
    this->m_S(4, 0) = -v[1];
    this->m_S(5, 0) = -v[2];

    this->m_S.getBases()[0].normalize();
}
}  // namespace PhysIKA