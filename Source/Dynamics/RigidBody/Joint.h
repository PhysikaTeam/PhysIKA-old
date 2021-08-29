#pragma once
#include "Framework/Framework/Node.h"
//#include "Core/Matrix/matrix_mxn.h"
//#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/JointSpace.h"
#include "Dynamics/RigidBody/Transform3d.h"
#include "Core/Vector/vector_3d.h"

#include <memory>

namespace PhysIKA {

/*!
    *    \class    Joint
    *    \brief    Joint constraint of TWO rigid bodies.
    *
    *    This class is base class of all joint class.
    *    A joint should include its structure information and joint space matrix.
    *
    */

//template<int Dof>
class Joint
{
    //DECLARE_CLASS_1(Joint, TDataType)
public:
    Joint(std::string name = "default");
    Joint(Node* predecessor, Node* successor);
    //Joint(const Joint& joint);

    void setRigidBody(Node* predecessor, Node* successor);

    Node* getParent()
    {
        return m_predecessor;
    }
    Node* getPredecessor()
    {
        return m_predecessor;
    }

    Node* getChild()
    {
        return m_successor;
    }
    Node* getSuccessor()
    {
        return m_successor;
    }

    //template<unsigned int Dof>
    //void setConstrainedSpaceMatrix(const JointSpace<float, Dof>& s) { this->m_S = s; }

    const Transform3d<float>& getXT() const
    {
        return m_XT;
    }
    void setXT(const Transform3d<float>& xt)
    {
        m_XT = xt;
    }

    const Transform3d<float>& getXJ() const
    {
        return m_XJ;
    }
    void setXJ(const Transform3d<float>& xj)
    {
        this->m_XJ = xj;
    }

    const Transform3d<float> getX() const
    {
        return m_XT * m_XJ;
    }

    // Get degree of freedom of joint
    // Should be overwritten
    virtual int getJointDOF() const
    {
        return 0;
    }

    // Get matrix of JointSpace. This function should be overwritten
    //template<int Dof>
    //virtual const JiontSpace<float,Dof>& getJointSpace()const { return JointSpace<float,Dof>(); }
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return JointSpace<float, 1>();
    }

    // Set matrix of JointSpace. This function should be overwritten
    //template<int Dof>
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        return;
    }  // { m_S = S; }

    virtual void update(double dt) {}

    /**
        * @brief Calculate relative position 
        */
    virtual void getRelativePostion(float* generalq, Vector3f& p, Quaternion<float>& qua) const {}

    virtual void getRelativeTransform(float* generalq, Transform3d<float>& trans) const {};

    virtual int violateConstraint(int i, float val)
    {
        if (!m_beConstraint[i] || (val >= m_lowerBound[i] && val <= m_upperBound[i]))
            return 0;
        else if (val < m_lowerBound[i])
            return -1;
        else
            return 1;
    }

    virtual void setConstraint(int i, float lower, float upper)
    {
        m_beConstraint[i] = true;
        m_lowerBound[i]   = lower;
        m_upperBound[i]   = upper;
    }

    /**
        * @brief Set relative joint positon (relative to predecessor)
        * @param 
        */
    virtual void setJointPosition(const Vector3f& relp, const Quaternion<float>& relqua)
    {
        m_relPos = relp;
        m_relQua = relqua;
        m_XT.set(-relqua.rotate(relp), m_relQua);
    }

    virtual float* getInitGeneralPos()
    {
        return m_initGeneralPos;
    }
    virtual const float* getInitGeneralPos() const
    {
        return m_initGeneralPos;
    }

    virtual inline void setMotorForce(float* force)
    {
        for (int i = 0; i < getJointDOF(); ++i)
            m_motorForce[i] = force[i];
    }
    inline const float* getMotorForce() const
    {
        return m_motorForce;
    }
    inline float* getMotorForce()
    {
        return m_motorForce;
    }

protected:
    //void _updateT();

public:
    virtual bool initialize()
    {
        return true;
    }

protected:
    Node* m_predecessor = 0;
    Node* m_successor   = 0;

    Vector3f          m_relPos;  // Joint postion relative to predecessor, in predecessor frame.
    Quaternion<float> m_relQua;  // Joint rotation relative to redecessor.

    // tree transformation. transformation from joint frame to predecessor frame.
    Transform3d<float> m_XT;

    // transformation from successor frame to joint frame.
    Transform3d<float> m_XJ;

    // we let the child class to define JointSpace matrix
    // as we don't know the Dof of the joint.
    //JointSpace<float, Dof> m_S;                    // constrained space matrix. defined in successor coordinate. 6*Dof

    bool  m_beConstraint[6] = { false, false, false, false, false, false };
    float m_lowerBound[6]   = { 0 };
    float m_upperBound[6]   = { 0 };

    float m_initGeneralPos[6] = { 0 };

    float m_motorForce[6] = { 0 };
};

//template<unsigned int Dof>
inline Joint::Joint(std::string name)  //:m_XT(6, 6), m_XJ(6, 6)
{
    //attachField(&m_predecessor, MechanicalState::mass(), "Total mass of the rigid body!", false);
    //m_XT.identity();
    //m_XJ.identity();
}

//template<unsigned int Dof>
inline Joint::Joint(Node* predecessor, Node* successor)
{
    m_predecessor = predecessor;
    m_successor   = successor;
}

//template<int Dof>
//inline Joint::Joint(const Joint& joint)
//{
//}

//template<unsigned int Dof>
inline void Joint::setRigidBody(Node* predecessor, Node* successor)
{
    this->m_predecessor = predecessor;
    this->m_successor   = successor;
}

}  // namespace PhysIKA