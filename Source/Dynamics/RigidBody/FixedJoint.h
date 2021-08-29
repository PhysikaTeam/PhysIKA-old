#pragma once

#include "Dynamics/RigidBody/Joint.h"

#include <memory>

namespace PhysIKA {
/*!
    *    \class    FixedJoint
    *
    *    Joint of 0 dof.
    *
    */

class FixedJoint : public Joint
{
    // The Base class is Joint<1>, not Joint<0>.
    // because it will cause error when JointSpace should not has a dof of 0.
    // Otherwise, JointSpace's menber m_data will be a 0 size array.

    //DECLARE_CLASS_1(FixedJoint, TDataType)
public:
    FixedJoint(std::string name = "default_fixed_joint");

    //FixedJoint(const FixedJoint& joint);

    FixedJoint(Node* predecessor, Node* successor);
    //virtual ~FixedJoint();

    // Dof is 0 not 1.
    virtual int getJointDOF() const
    {
        return 0;
    }

private:
    JointSpace<float, 1> m_S;
};

//#ifdef PRECISION_FLOAT
//    template class FixedJoint<DataType3f>;
//#else
//    template class FixedJoint<DataType3d>;
//#endif
}  // namespace PhysIKA