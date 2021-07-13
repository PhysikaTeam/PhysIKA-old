#pragma once
#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/Joint.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Core/Quaternion/quaternion.h"
#include "Framework/Topology/Frame.h"
//#include "Rendering/SurfaceMeshRender.h"
#include "Framework/Mapping/FrameToPointSet.h"
#include "Framework/Topology/TriangleSet.h"
//#include "Rendering/RigidMeshRender.h"
//#include "Rendering/SurfaceMeshRender.h"
#include "Inertia.h"

#include <string>

namespace PhysIKA {

/*!
    *    \class    RigidBody
    *    \brief    Rigid body dynamics.
    *
    *    This class implements a simple rigid body.
    *
    */
template <typename TDataType>
class RigidBody2 : public Node
{
    DECLARE_CLASS_1(RigidBody2, TDataType)
public:
    RigidBody2(std::string name = "default");
    virtual ~RigidBody2();

    virtual void loadShape(std::string filename);

    void addChildJoint(std::shared_ptr<Joint> child_joint);
    //void addChildJoint(Joint* child_joint);
    //void setParentJoint(std::shared_ptr<Joint> parent_joint);
    void setParentJoint(Joint* parent_joint);

    const ListPtr<Joint>& getChildJoints()
    {
        return m_child_joints;
    }

    Joint* getParentJoint() const
    {
        return m_parent_joint;
    }
    const ListPtr<Joint>& getChildJoints() const
    {
        return m_child_joints;
    }
    //ListPtr<Joint<TDataType>>& getChildJoints() { return m_child_joints; }

    void setParent(Node* p);

    const Inertia<float>& getI()
    {
        return m_I;
    }

    void setI(const Inertia<float>& I)
    {
        m_I = I;
        this->setMass(I.getMass());
    }

    // get all descendant nodes
    // virtual std::vector<std::shared_ptr<RigidBody2<TDataType>>> getAllNode();
    // virtual void getAllNode(std::vector<std::shared_ptr<RigidBody2<TDataType>>>& all_node);

    // get all descendant nodes in pair
    // pair: first: parent id in the vector; second: shared_ptr of node
    //virtual const std::vector< std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& getAllParentidNodePair()const;
    //virtual void getAllParentidNodePair(std::vector< std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& all_node);

    virtual void setGeometrySize(float sx, float sy, float sz)
    {
        m_sizex = sx;
        m_sizey = sy;
        m_sizez = sz;
    }

    Vector3f getGlobalR() const
    {
        return m_global_r;
    }
    Quaternion<float> getGlobalQ() const
    {
        return m_global_q;
    }

    void setGlobalR(const Vector3f& r)
    {
        m_global_r = r;
    }
    void setGlobalQ(const Quaternion<float>& q)
    {
        m_global_q = q;
    }

    const Vector3f& getRelativeR() const
    {
        return m_relativePos;
    }
    Vector3f& getRelativeR()
    {
        return m_relativePos;
    }
    void setRelativeR(const Vector3f& relpos)
    {
        m_relativePos = relpos;
    }

    const Quaternion<float>& getRelativeQ() const
    {
        return m_relativeRot;
    }
    Quaternion<float>& getRelativeQ()
    {
        return m_relativeRot;
    }
    void setRelativeQ(const Quaternion<float>& qua)
    {
        m_relativeRot = qua;
    }

    void setLinearVelocity(const Vector3f& linv)
    {
        m_globalLinVel = linv;
    }
    const Vector3f& getLinearVelocity() const
    {
        return m_globalLinVel;
    }
    Vector3f& getLinearVelocity()
    {
        return m_globalLinVel;
    }

    void setAngularVelocity(const Vector3f& angv)
    {
        m_globalAngVel = angv;
    }
    const Vector3f& getAngularVelocity() const
    {
        return m_globalAngVel;
    }
    Vector3f& getAngularVelocity()
    {
        return m_globalAngVel;
    }

    void addExternalForce(const Vector3f& force)
    {
        m_externalForce += force;
    }
    void setExternalForce(const Vector3f& force)
    {
        m_externalForce = force;
    }
    const Vector3f& getExternalForce() const
    {
        return m_externalForce;
    }
    Vector3f& getExternalForce()
    {
        return m_externalForce;
    }

    void addExternalTorque(const Vector3f& torque)
    {
        m_externalTorque += torque;
    }
    void setExternalTorque(const Vector3f& torque)
    {
        m_externalTorque = torque;
    }
    const Vector3f& getExternalTorque() const
    {
        return m_externalTorque;
    }
    Vector3f& getExternalTorque()
    {
        return m_externalTorque;
    }

    inline std::shared_ptr<Frame<TDataType>> getTransformationFrame()
    {
        return m_frame;
    }

    float getRadius() const
    {
        return m_boundingRadius;
    }
    void setRadius(float radius)
    {
        m_boundingRadius = radius;
    }

    void advance(Real dt) override;
    void updateTopology() override;

    int getId() const
    {
        return m_id;
    }
    void setId(int id)
    {
        m_id = id;
    }

    float getMu() const
    {
        return m_mu;
    }
    void setMu(float mu)
    {
        m_mu = mu;
    }

    float getRho() const
    {
        return m_rho;
    }
    void setRho(float rho)
    {
        m_rho = rho;
    }

    //void setActive

    void setLinearDamping(float damp)
    {
        m_linearDamping = damp;
    }
    float getLinearDamping()
    {
        return m_linearDamping;
    }
    void setAngularDamping(float damp)
    {
        m_angularDamping = damp;
    }
    float getAngularDamping()
    {
        return m_angularDamping;
    }

    //void integrateForce(Real dt);

    int getCollisionFilterGroup()
    {
        return m_collisionGroup;
    }
    void setCollisionFilterGroup(int group)
    {
        m_collisionGroup = group;
    }

    int getCollisionFilterMask()
    {
        return m_collisionMask;
    }
    void setCollisionFilterMask(int mask)
    {
        m_collisionMask = mask;
    }

public:
    bool initialize() override;

private:
    Joint*         m_parent_joint;  // the parent joint
    ListPtr<Joint> m_child_joints;  // list of child joints

    int m_id = -1;

    // Physical property info
    Inertia<float> m_I;  // inertia
    float          m_mu  = 1.0;
    float          m_rho = 1.0;

    // Geometric info
    float  m_sizex        = 1.0;
    float  m_sizey        = 1.0;
    float  m_sizez        = 1.0;
    double m_global_scale = 1.0;

    // state info
    Vector3f          m_global_r;
    Quaternion<float> m_global_q;

    Vector3f          m_relativePos;
    Quaternion<float> m_relativeRot;
    Vector3f          m_globalLinVel;
    Vector3f          m_globalAngVel;

    Vector3f m_externalForce;
    Vector3f m_externalTorque;

    // others
    std::shared_ptr<TriangleSet<TDataType>> m_triSet;
    std::shared_ptr<Frame<TDataType>>       m_frame;
    //std::shared_ptr<RigidMeshRender> m_render;
    //std::shared_ptr<FrameToPointSet<TDataType>> m_surfaceMapping;

    float m_boundingRadius = 0;

    float m_linearDamping  = 0.5;
    float m_angularDamping = 0.5;

    //bool beActive = true;

    // Test property
    int m_collisionGroup = 1;  //
    int m_collisionMask  = 1;  //
};

struct RigidInteractionInfo
{
    Vector3f          position;
    Quaternion<float> rotation;

    Vector3f linearVelocity;
    Vector3f angularVelocity;
};

//template<typename TDataType>
//void PhysIKA::RigidBody2<TDataType>::integrateForce(Real dt)
//{
//    linVelocity += m_externalForce * invMass * dt;

//    //if(invInertia[0]==0 || invInertia[1] != 0 && invInertia[1] != 0)
//    Vector<Real, 3> angv = angVelocity;
//    pose.invRotate(angv);        // to local.
//    Vector<Real, 3> omegaI(invInertia[0] == 0 ? 0 : angv[0] / invInertia[0],
//        invInertia[1] == 0 ? 0 : angv[1] / invInertia[1],
//        invInertia[2] == 0 ? 0 : angv[2] / invInertia[2]);
//    omegaI = angv.cross(omegaI);
//    Vector<Real, 3> extT = externalTorque;
//    pose.invRotate(extT);
//    omegaI = (extT - omegaI)*invInertia * dt;
//    angVelocity = angv + omegaI;
//}

#ifdef PRECISION_FLOAT
template class RigidBody2<DataType3f>;
typedef std::shared_ptr<RigidBody2<DataType3f>> RigidBody2_ptr;

#else
template class RigidBody2<DataType3d>;
typedef std::shared_ptr<RigidBody2<DataType3d>> RigidBody2_ptr;

#endif
}  // namespace PhysIKA