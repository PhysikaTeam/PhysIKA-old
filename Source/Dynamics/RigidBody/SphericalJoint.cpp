#include "SphericalJoint.h"
//#include "Dynamics/RigidBody/RigidUtil.h"

namespace PhysIKA {

SphericalJoint::SphericalJoint(std::string name)
    : Joint(name)
{
}

SphericalJoint::SphericalJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void SphericalJoint::setJointInfo(const Vector3f& r)
{
    for (int i = 0; i < 3; ++i)
    {
        Vector3f unit_axis;
        unit_axis[i] = 1;

        Vector3f v = unit_axis.cross(r);

        this->m_S(0, i) = unit_axis[0];
        this->m_S(1, i) = unit_axis[1];
        this->m_S(2, i) = unit_axis[2];
        this->m_S(3, i) = -v[0];
        this->m_S(4, i) = -v[1];
        this->m_S(5, i) = -v[2];

        this->m_S.getBases()[i].normalize();
    }
}

}  // namespace PhysIKA