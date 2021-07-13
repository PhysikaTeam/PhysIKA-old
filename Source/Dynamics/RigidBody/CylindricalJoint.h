#pragma once
#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/Joint.h"
//#include "Dynamics/RigidBody/RigidBody2.h"
#include <memory>

namespace PhysIKA {
/**
    Cylindrical joint. It can rotate and move along a certain axis.

    Note: the center of successor should be on the rotation axis.
    */

class CylindricalJoint : public Joint
{
    //DECLARE_CLASS_1(Joint, TDataType)
public:
    CylindricalJoint(std::string name = "default");

    CylindricalJoint(Node* predecessor, Node* successor);

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 2;
    }
    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }
    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 2>&>(S);
    }  // { m_S = S; }

    // set informations
    // axis: rotation axis, dim = 3
    void setJointInfo(const VectorBase<float>& axis);

private:
    JointSpace<float, 2> m_S;  // constrained space matrix. defined in successor coordinate. 6*Dof
};

//
//#ifdef PRECISION_FLOAT
//    template class Joint<DataType3f>;
//#else
//    template class Joint<DataType3d>;
//#endif
}  // namespace PhysIKA