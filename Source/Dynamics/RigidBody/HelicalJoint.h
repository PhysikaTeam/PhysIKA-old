#pragma once
#include "Dynamics/RigidBody/Joint.h"

#include <memory>

namespace PhysIKA {
/**
    Helical joint. ( Screw joint)

    Note: the center of successor should be on the rotation axis
    */

class HelicalJoint : public Joint
{
    //DECLARE_CLASS_1(Joint, TDataType)
public:
    HelicalJoint(std::string name = "default");
    HelicalJoint(Node* predecessor, Node* successor);

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 1;
    }
    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }
    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 1>&>(S);
    }  // { m_S = S; }

    // set informations
    // axis: rotation axis, dim = 3
    // h: the motion rate (m/randian) that the successor will move when it rotate
    void setJointInfo(const VectorBase<float>& axis, float h);

private:
    JointSpace<float, 1> m_S;
};

//
//#ifdef PRECISION_FLOAT
//    template class Joint<DataType3f>;
//#else
//    template class Joint<DataType3d>;
//#endif
}  // namespace PhysIKA