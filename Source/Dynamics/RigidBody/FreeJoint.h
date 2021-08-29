#pragma once

#include "Dynamics/RigidBody/Joint.h"

//#include "Dynamics/RigidBody/RigidBody2.h"
#include <memory>

namespace PhysIKA {

/*!
    *    \class    FreeJoint
    *    \brief    6 dof joint.
    *
    *    This kind of joint has no constraint on rigid bodies.
    *    But it is useful for manage the system.
    *
    */

class FreeJoint : public Joint
{
    //DECLARE_CLASS_1(FreeJoint, TDataType)
public:
    FreeJoint(std::string name = "default");

    FreeJoint(Node* predecessor, Node* successor);
    //virtual ~FreeJoint();

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 6;
    }
    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }
    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 6>&>(S);
    }  // { m_S = S; }

private:
    JointSpace<float, 6> m_S;
};

//#ifdef PRECISION_FLOAT
//    template class FreeJoint<DataType3f>;
//#else
//    template class FreeJoint<DataType3d>;
//#endif
}  // namespace PhysIKA