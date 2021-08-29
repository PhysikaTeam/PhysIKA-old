#include "Joint.h"
//#include "Dynamics/RigidBody/RigidUtil.h"
#include "PlanarJoint.h"
#include "Core/Vector/vector_3d.h"
namespace PhysIKA {
//IMPLEMENT_CLASS_1(Joint, TDataType)

PlanarJoint::PlanarJoint(std::string name)
    : Joint(name)
{
}

PlanarJoint::PlanarJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void PlanarJoint::setJointInfo(const Vector3f& plane_norm)
{
    Vector3f unit_norm(plane_norm[0], plane_norm[1], plane_norm[2]);
    unit_norm.normalize();

    // rotation
    m_S(0, 0) = unit_norm[0];
    m_S(1, 0) = unit_norm[1];
    m_S(2, 0) = unit_norm[2];

    // axis 1
    Vector3f axis1;
    int      n0_axis         = (unit_norm[0] != 0) ? 0 : ((unit_norm[1] != 0) ? 1 : 2);  ///< index of none 0 value axis
    axis1[n0_axis]           = unit_norm[(n0_axis + 1) % 3];
    axis1[(n0_axis + 1) % 3] = -unit_norm[n0_axis];
    axis1[(n0_axis + 2) % 3] = 0;
    axis1.normalize();

    // tanslation 1
    m_S(3, 1) = axis1[0];
    m_S(4, 1) = axis1[1];
    m_S(5, 1) = axis1[2];

    // axis 2
    Vector3f axis2;
    axis2 = unit_norm.cross(axis1).normalize();

    // tanslation 1
    m_S(3, 2) = axis2[0];
    m_S(4, 2) = axis2[1];
    m_S(5, 2) = axis2[2];
}
}  // namespace PhysIKA