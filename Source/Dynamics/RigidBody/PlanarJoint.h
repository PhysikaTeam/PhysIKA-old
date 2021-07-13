#pragma once
#include "Dynamics/RigidBody/Joint.h"
//#include "Dynamics/RigidBody/RigidBody2.h"
#include <memory>

namespace PhysIKA {
/**
    Planar joint. It can move freely in a certain plane.

    */

class PlanarJoint : public Joint
{
    //DECLARE_CLASS_1(Joint, TDataType)
public:
    PlanarJoint(std::string name = "default");

    PlanarJoint(Node* predecessor, Node* successor);
    //virtual ~PlanarJoint();

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 3;
    }
    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }
    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 3>&>(S);
    }  // { m_S = S; }

    // set informations
    // plane_norm: normal of the plan
    void setJointInfo(const Vector3f& plane_norm);

private:
    JointSpace<float, 3> m_S;
};

//
//#ifdef PRECISION_FLOAT
//    template class Joint<DataType3f>;
//#else
//    template class Joint<DataType3d>;
//#endif
}  // namespace PhysIKA