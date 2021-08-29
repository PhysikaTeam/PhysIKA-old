#include "RigidBody2.h"

#include "Core/Typedef.h"
#include <queue>
#include <random>
#include <ctime>
#include "Dynamics/RigidBody/RigidTimeIntegrationModule.h"
//#include "Rendering/RigidMeshRender.h"
//#include "Rendering/SurfaceMeshRender.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(RigidBody2, TDataType)

template <typename TDataType>
PhysIKA::RigidBody2<TDataType>::RigidBody2(std::string name)
    : Node(name)  //,m_I(6,6)
    //,m_X(6,6), m_v(6), m_a(6), m_external_f(6), m_r(6)
    , m_global_r(0, 0, 0)
    , m_global_q(0, 0, 0, 1)
    , m_globalLinVel(0, 0, 0)
    , m_globalAngVel(0, 0, 0)
{
    m_parent_joint = 0;
    m_triSet       = 0;
    m_frame        = std::make_shared<Frame<TDataType>>();

    this->setActive(true);
    this->setVisible(true);
}

template <typename TDataType>
RigidBody2<TDataType>::~RigidBody2()
{
}

template <typename TDataType>
void RigidBody2<TDataType>::loadShape(std::string filename)
{
    m_triSet = TypeInfo::cast<TriangleSet<DataType3f>>(this->getTopologyModule());

    if (!m_triSet)
    {
        m_triSet = std::make_shared<TriangleSet<TDataType>>();
        this->setTopologyModule(m_triSet);
    }
    std::shared_ptr<TriangleSet<TDataType>> surface = m_triSet;  // TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule());
    surface->loadObjFile(filename);
}

template <typename TDataType>
void RigidBody2<TDataType>::addChildJoint(std::shared_ptr<Joint> child_joint)
{
    m_child_joints.push_back(child_joint);
}

template <typename TDataType>
void RigidBody2<TDataType>::setParentJoint(Joint* parent_joint)
{
    m_parent_joint = parent_joint;
    m_parent_joint->setRigidBody(m_parent, this);
}

template <typename TDataType>
void RigidBody2<TDataType>::setParent(Node* p)
{
    Node::setParent(p);

    if (m_parent_joint)
        m_parent_joint->setRigidBody(p, this);
}

template <typename TDataType>
void RigidBody2<TDataType>::updateTopology()
{
    m_frame->setCenter(m_global_r * m_global_scale);
    m_frame->setOrientation(m_global_q.get3x3Matrix());
    //m_surfaceMapping->apply();

    //float rad;
    //Vector3f rotate_axis;
    //m_global_q.getRotation(rad, rotate_axis);
    //Quaternion<float> rotation(rad, rotate_axis[0], rotate_axis[1], rotate_axis[2]);
    //m_render->setRotation(rotation);
    //m_render->setTranslatioin(m_global_r * m_global_scale);

    //m_render->setTriangleRotation(m_global_q);
    //m_render->setTriangleTranslation(m_global_r * m_global_scale);
}

template <typename TDataType>
bool RigidBody2<TDataType>::initialize()
{
    //if (!m_triSet)
    //{
    //    m_triSet = std::make_shared<PhysIKA::TriangleSet<TDataType>>();
    //    m_triSet->loadObjFile("../../Media/standard/standard_cube.obj");
    //    this->setTopologyModule(m_triSet);
    //}
    //m_triSet = TypeInfo::cast<TriangleSet<DataType3f>>(this->getTopologyModule());

    //if (m_triSet)
    //{
    //    m_triSet->translate(Vector3f(0.0, 0.0, 0.0));
    //    //m_triSet->scale(Vector3f(m_sizex, m_sizey, m_sizez)* m_global_scale / 2.0);
    //}

    this->updateTopology();

    return true;
}

template <typename TDataType>
void RigidBody2<TDataType>::advance(Real dt)
{
    return;

    m_globalLinVel *= pow(1.0 - m_linearDamping, dt);
    m_globalAngVel *= pow(1.0 - m_angularDamping, dt);

    //std::cout << "  ** Rigid Pos: " << m_global_r[0] << "  " << m_global_r[1] << "  " << m_global_r[2] << std::endl;
    if (m_parent_joint)
        m_parent_joint->update(dt);
}
}  // namespace PhysIKA