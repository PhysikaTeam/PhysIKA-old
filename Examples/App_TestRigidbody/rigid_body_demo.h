#pragma once

#include <iostream>

#include "Dynamics/RigidBody/urdf.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"
//#include "Dynamics/RigidBody/RigidBody2.h"
//#include "Dynamics/RigidBody/Joint.h"
#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "GUI/GlutGUI/GLApp.h"
#include <string>
#include <queue>
#include "Dynamics/RigidBody/RigidUtil.h"

#include "Dynamics/RigidBody/FixedJoint.h"
#include "Dynamics/RigidBody/RevoluteJoint.h"

#include "Core/Vector/vector_2d.h"

#include "Dynamics/RigidBody/BroadPhaseDetector.h"

#include "Dynamics/RigidBody/PlanarJoint.h"
#include "Dynamics/RigidBody/PrismaticJoint.h"
#include "Dynamics/RigidBody/HelicalJoint.h"
#include "Dynamics/RigidBody/CylindricalJoint.h"
#include "Dynamics/RigidBody/FreeJoint.h"
#include "Dynamics/RigidBody/SphericalJoint.h"

#include "Rendering/RigidMeshRender.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <algorithm>

#include <random>

using namespace std;

using namespace PhysIKA;

void outTree(shared_ptr<RigidBody2<DataType3f>> node)
{
    if (!node)
    {
        return;
    }

    cout << node->getName().c_str() << endl;
    ListPtr<Joint> child_joints = node->getChildJoints();

    for (auto iter = child_joints.begin(); iter != child_joints.end(); ++iter)
    {
        cout << "Parent: " << (*iter)->getParent()->getName().c_str() << endl;
        cout << "Child:  " << (*iter)->getChild()->getName().c_str() << endl;
    }

    ListPtr<Node>& children = node->getChildren();
    for (auto iter = children.begin(); iter != children.end(); ++iter)
    {
        outTree(dynamic_pointer_cast<RigidBody2<DataType3f>>(*iter));
    }
}

void demoLoadFile()
{
    //srand(time(0));
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    string                                filename("D:\\Projects\\Physika\\PhysiKA_Rigid\\Media\\urdf\\test_robot2.urdf");
    Urdf                                  urdf;
    shared_ptr<RigidBodyRoot<DataType3f>> rigid_root(urdf.loadFile(filename.c_str()));
    root->addChild(rigid_root);

    Vector3f g(0, -0, 0);  //g[4] = -5;
    rigid_root->setGravity(g);

    auto                              all_rigid = rigid_root->getAllParentidNodePair();
    std::vector<SpatialVector<float>> v(all_rigid.size(), SpatialVector<float>());

    SystemMotionState&      state   = *(rigid_root->getSystemState()->m_motionState);
    const std::vector<int>& idx_map = rigid_root->getJointIdxMap();

    for (int i = 0; i < all_rigid.size(); ++i)
    {
        int parent_id = all_rigid[i].first;

        if (parent_id >= 0)
        {
            state.generalVelocity[idx_map[i]] = 1.5;

            auto cur_node = all_rigid[i].second;
            state.m_v[i]  = (cur_node->getParentJoint()->getJointSpace().mul(&(state.generalVelocity[idx_map[i]])));
        }
    }

    outTree(rigid_root);

    GLApp window;
    window.createWindow(1024, 768);

    window.mainLoop();
}

void demo_PlanarJoint()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    /// root
    RigidBodyRoot_ptr rigid_root = std::make_shared<RigidBodyRoot<DataType3f>>("rigid_root");
    root->addChild(rigid_root);
    //rigid_root->setGravity(Vector3f(0, -9, 5));

    /// system state
    std::shared_ptr<SystemState>       system_state = rigid_root->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    /// rigid body 1
    RigidBody2_ptr rigid1 = std::make_shared<RigidBody2<DataType3f>>("rigid1");
    rigid_root->addChild(rigid1);

    /// rigid body 2
    RigidBody2_ptr rigid2 = std::make_shared<RigidBody2<DataType3f>>("rigid2");
    rigid_root->addChild(rigid2);

    /// rigid body 3
    RigidBody2_ptr rigid3 = std::make_shared<RigidBody2<DataType3f>>("rigid3");
    rigid1->addChild(rigid3);

    /// joint 1. rigid_roott -> rigid1
    std::shared_ptr<FixedJoint> joint1(new FixedJoint("from_base1"));
    joint1->setRigidBody(rigid_root.get(), rigid1.get());
    rigid_root->addChildJoint(joint1);
    rigid1->setParentJoint(joint1.get());

    /// joint 2. rigid_root -> rigid2
    std::shared_ptr<FixedJoint> base_joint2(new FixedJoint("from_base2"));
    base_joint2->setRigidBody(rigid_root.get(), rigid2.get());
    rigid_root->addChildJoint(base_joint2);
    rigid2->setParentJoint(base_joint2.get());

    /// joint 3. rigid1 -> rigid3
    std::shared_ptr<PlanarJoint> planar_joint(new PlanarJoint("planar_joint"));
    planar_joint->setRigidBody(rigid1.get(), rigid3.get());
    rigid1->addChildJoint(planar_joint);
    rigid3->setParentJoint(planar_joint.get());

    /// update tree info
    rigid_root->updateTree();

    /// setup rigid bodies and joints properties.
    {
        float          box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
        float          rigid_mass = 12;
        float          ixx = 10.0, iyy = 1.0, izz = 10.0;
        Vector3f       ixyz(10.0, 1.0, 10.0);
        Inertia<float> rigid_inertia(rigid_mass, ixyz);

        /// **************** joint 3
        Vector3f joint_r3;
        joint_r3[1] = 1.0;
        Quaternion<float>  joint_q3;
        Transform3d<float> joint_X3(joint_r3, joint_q3.getConjugate());

        Vector3f axis_norm;
        axis_norm[1] = 1.0;
        planar_joint->setJointInfo(axis_norm);

        /// ******** rigid 1
        int id1 = rigid1->getId();

        /// position and rotaion of rigid1
        Vector3f rigid_r1;
        rigid_r1[1] = 2;
        Quaternion<float>  rigid_q1(-0.5, Vector3f(0, 0, 1));
        Transform3d<float> rigid_X1(rigid_r1, rigid_q1.getConjugate());
        motion_state->m_rel_r[id1] = rigid_r1;
        motion_state->m_rel_q[id1] = rigid_q1;
        motion_state->m_X[id1]     = rigid_X1;

        rigid1->setGeometrySize(15, 1, 5);

        rigid1->setMass(rigid_mass);
        rigid1->setI(rigid_inertia);

        /// ************ Rigid 2
        int id2 = rigid2->getId();

        Vector3f rigid_r2;
        rigid_r2[1] = 4;
        Quaternion<float>  rigid_q2(-0.5, Vector3f(0, 0, 1));
        Transform3d<float> rigid_X2(rigid_r2, rigid_q2.getConjugate());
        motion_state->m_rel_r[id2] = rigid_r2;
        motion_state->m_rel_q[id2] = rigid_q2;
        motion_state->m_X[id2]     = rigid_X2;

        rigid2->setGeometrySize(15, 1, 5);
        rigid2->setMass(rigid_mass);
        rigid2->setI(rigid_inertia);

        /// *************** rigid 3, dynamic
        int id3 = rigid3->getId();

        Vector3f          rigid_r3;
        Quaternion<float> rigid_q3;
        //Transform3d<float> rigid_X3(rigid_r3, rigid_q3.getConjugate());
        motion_state->m_rel_r[id3] = joint_r3 + joint_q3.rotate(rigid_r3);
        motion_state->m_rel_q[id3] = joint_q3 * rigid_q3;
        motion_state->m_X[id3]     = Transform3d<float>(motion_state->m_rel_r[id3], motion_state->m_rel_q[id3].getConjugate());

        rigid3->setGeometrySize(1, 1, 1);
        rigid3->setMass(rigid_mass);
        rigid3->setI(Inertia<float>(rigid_mass, Vector3f(2, 2, 2)));

        /// set velocity
        SpatialVector<float> relv(1, 1, 1, -5, 0, 5);                                                       ///< relative velocity, in successor frame.
        planar_joint->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[0]));              ///< map relative velocity into joint space vector.
        motion_state->m_v[id3] = (planar_joint->getJointSpace().mul(&(motion_state->generalVelocity[0])));  ///<

        //motion_state->m_dq[0] = -15;
        //motion_state->m_dq[1] = -0;
        //motion_state->m_dq[2] = 5;

        //motion_state->m_v[id3] = (planar_joint->getJointSpace().mul(&(motion_state->m_dq[0])));

        motion_state->updateGlobalInfo();
    }

    //rigid_root->initTreeGlobalInfo();

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

void demo_PrismaticJoint()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    /// root
    RigidBodyRoot_ptr rigid_root = std::make_shared<RigidBodyRoot<DataType3f>>("rigid_root");
    root->addChild(rigid_root);
    //rigid_root->setGravity(Vector3f(0, -9, 5));

    /// system state
    std::shared_ptr<SystemState>       system_state = rigid_root->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    /// rigid body 1
    RigidBody2_ptr rigid1 = std::make_shared<RigidBody2<DataType3f>>("rigid1");
    rigid_root->addChild(rigid1);

    /// rigid body 2
    RigidBody2_ptr rigid2 = std::make_shared<RigidBody2<DataType3f>>("rigid2");
    rigid1->addChild(rigid2);

    /// joint . rigid_root -> rigid1
    std::shared_ptr<FixedJoint> joint1(new FixedJoint("from_base"));
    joint1->setRigidBody(rigid_root.get(), rigid1.get());
    rigid_root->addChildJoint(joint1);
    rigid1->setParentJoint(joint1.get());

    /// joint . rigid1 -> rigid2
    std::shared_ptr<PrismaticJoint> joint2(new PrismaticJoint("planar_joint"));
    joint2->setRigidBody(rigid1.get(), rigid2.get());
    rigid1->addChildJoint(joint2);
    rigid2->setParentJoint(joint2.get());

    /// update tree info
    rigid_root->updateTree();

    {
        float box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
        float rigid_mass = 12;
        float ixx = 10.0, ixy = 0.0, ixz = 0.0, iyy = 1.0, iyz = 0.0, izz = 10.0;

        /// ********* prismatic joint info
        Vector3f          joint_r2;  //joint_r2[1] = 1.0;
        Quaternion<float> joint_q2;
        Vector3f          axis_norm;
        axis_norm[1] = 1.0;
        joint2->setJointInfo(axis_norm);

        /// ********** rigid 1
        int id1 = rigid1->getId();

        Vector3f rigid_r1;
        rigid_r1[1] = 6;
        Quaternion<float> rigid_q1;
        rigid_q1.set(Vector3f(-0.5, -0.0, 0));
        //Transform3d<float> rigid_X1;
        motion_state->m_rel_r[id1] = rigid_r1;
        motion_state->m_rel_q[id1] = rigid_q1;
        rigid1->setGeometrySize(2, 2, 2);
        rigid1->setI(Inertia<float>(rigid_mass, Vector3f(8, 8, 8)));

        /// ************** rigid 2
        int id2 = rigid2->getId();

        Vector3f          rigid_r2;
        Quaternion<float> rigid_q2;
        motion_state->m_rel_r[id2] = joint_r2 + joint_q2.rotate(rigid_r2);
        motion_state->m_rel_q[id2] = joint_q2 * rigid_q2;
        rigid2->setGeometrySize(1, 7, 1);
        rigid2->setI(Inertia<float>(rigid_mass, Vector3f(50, 2, 50)));

        /// set velocity
        SpatialVector<float> relv(1, 1, 1, -5, 5, 5);                                                 ///< relative velocity, in successor frame.
        joint2->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[0]));              ///< map relative velocity into joint space vector.
        motion_state->m_v[id2] = (joint2->getJointSpace().mul(&(motion_state->generalVelocity[0])));  ///< set

        motion_state->updateGlobalInfo();
    }

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

void demo_middleAxis()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    /// root
    RigidBodyRoot_ptr rigid_root = std::make_shared<RigidBodyRoot<DataType3f>>("rigid_root");
    root->addChild(rigid_root);
    rigid_root->setGravity(Vector3f(0, -0, 0));

    /// system state
    std::shared_ptr<SystemState>       system_state = rigid_root->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    Vector3f rigid_r[3];
    rigid_r[0] = Vector3f(-3, 6, 0);
    rigid_r[1] = Vector3f(0, 6, 0);
    rigid_r[2] = Vector3f(3, 6, 0);

    SpatialVector<float> rigid_v[3];
    rigid_v[0] = SpatialVector<float>(0.1, 5, 0, 0, 0, 0);
    rigid_v[1] = SpatialVector<float>(5, 0, 0.1, 0, 0, 0);
    rigid_v[2] = SpatialVector<float>(0, 0.1, 5, 0, 0, 0);
    //Vector3f rigid_w[3];
    //rigid_w[0] = Vector3f(0.1, 5, 0);
    //rigid_w[0] = Vector3f(5, 0, 0.1);
    //rigid_w[0] = Vector3f(0, 0.1, 5);

    RigidBody2_ptr              rigid1[3];
    RigidBody2_ptr              rigid2[3];
    std::shared_ptr<FreeJoint>  joint1[3];
    std::shared_ptr<FixedJoint> fixed_joint[3];

    for (int i = 0; i < 3; ++i)
    {
        /// rigid body 1
        rigid1[i] = std::make_shared<RigidBody2<DataType3f>>("rigid1");
        rigid_root->addChild(rigid1[i]);

        /// rigid body 2
        rigid2[i] = std::make_shared<RigidBody2<DataType3f>>("rigid2");
        rigid1[i]->addChild(rigid2[i]);

        /// joint . rigid_root -> rigid1
        joint1[i] = std::make_shared<FreeJoint>("from_base");
        joint1[i]->setRigidBody(rigid_root.get(), rigid1[i].get());
        rigid_root->addChildJoint(joint1[i]);
        rigid1[i]->setParentJoint(joint1[i].get());

        /// joint . rigid1 -> rigid2
        fixed_joint[i] = std::make_shared<FixedJoint>("fixed_joint");
        fixed_joint[i]->setRigidBody(rigid1[i].get(), rigid2[i].get());
        rigid1[i]->addChildJoint(fixed_joint[i]);
        rigid2[i]->setParentJoint(fixed_joint[i].get());
    }
    /// update tree info
    rigid_root->updateTree();

    auto idx_map = rigid_root->getJointIdxMap();

    for (int i = 0; i < 3; ++i)
    {
        float box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
        float rigid_mass = 12;
        float ixx = 10.0, ixy = 0.0, ixz = 0.0, iyy = 1.0, iyz = 0.0, izz = 10.0;

        /// ********* free joint info
        //Vector3f joint_r1

        /// ********* fixed joint info
        Vector3f          joint_r2(0.25, 0, 0);
        Quaternion<float> joint_q2;

        /// ********** rigid 1
        int id1 = rigid1[i]->getId();

        //Vector3f rigid_r1;
        //rigid_r1[1] = 6;
        Quaternion<float> rigid_q1;  //rigid_q1.set(Vector3f(-0.5, -0.0, 0));
        motion_state->m_rel_r[id1] = rigid_r[i];
        motion_state->m_rel_q[id1] = rigid_q1;
        rigid1[i]->setGeometrySize(0.5, 2, 0.5);
        rigid1[i]->setI(Inertia<float>(rigid_mass, Vector3f(4.25, 0.5, 4.25)));

        /// ************** rigid 2
        int id2 = rigid2[i]->getId();

        Vector3f rigid_r2(0.5, 0, 0);
        rigid_r2 = joint_r2 + joint_q2.rotate(rigid_r2);
        Quaternion<float> rigid_q2;
        motion_state->m_rel_r[id2] = rigid_r2;
        motion_state->m_rel_q[id2] = joint_q2 * rigid_q2;
        rigid2[i]->setGeometrySize(1, 0.5, 0.5);
        rigid2[i]->setI(Inertia<float>(rigid_mass / 2, Vector3f(0.25, 0.625, 0.625)));

        /// set velocity
        //Vector3f center = rigid_r[i] + rigid_r2 *(1.0 / 3.0);
        Transform3d<float>   trans_center(-rigid_r2 * (1.0 / 3.0), rigid_q1.getConjugate());
        SpatialVector<float> relv = trans_center.transformM(rigid_v[i]);                                            ///< relative velocity, in successor frame.
        joint1[i]->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[idx_map[id1]]));              ///< map relative velocity into joint space vector.
        motion_state->m_v[id1] = (joint1[i]->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id1]])));  ///< set
    }

    motion_state->updateGlobalInfo();

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

void demo_SphericalJoint()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    /// root
    RigidBodyRoot_ptr rigid_root = std::make_shared<RigidBodyRoot<DataType3f>>("rigid_root");
    root->addChild(rigid_root);
    //rigid_root->setGravity(Vector3f(0, 0, 0));

    /// system state
    std::shared_ptr<SystemState>       system_state = rigid_root->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    /// rigid body 1
    RigidBody2_ptr rigid1 = std::make_shared<RigidBody2<DataType3f>>("rigid1");
    rigid_root->addChild(rigid1);

    /// rigid body 2
    RigidBody2_ptr rigid2 = std::make_shared<RigidBody2<DataType3f>>("rigid2");
    rigid1->addChild(rigid2);

    /// rigid body 3
    RigidBody2_ptr rigid3 = std::make_shared<RigidBody2<DataType3f>>("rigid3");
    rigid2->addChild(rigid3);

    /// joint . rigid_root -> rigid1
    std::shared_ptr<SphericalJoint> joint1(new SphericalJoint("joint1"));
    joint1->setRigidBody(rigid_root.get(), rigid1.get());
    rigid_root->addChildJoint(joint1);
    rigid1->setParentJoint(joint1.get());

    /// joint . rigid1 -> rigid2
    std::shared_ptr<SphericalJoint> joint2(new SphericalJoint("joint2"));
    joint2->setRigidBody(rigid1.get(), rigid2.get());
    rigid1->addChildJoint(joint2);
    rigid2->setParentJoint(joint2.get());

    /// joint . rigid2 -> rigid3
    std::shared_ptr<SphericalJoint> joint3(new SphericalJoint("joint2"));
    joint3->setRigidBody(rigid2.get(), rigid3.get());
    rigid2->addChildJoint(joint3);
    rigid3->setParentJoint(joint3.get());

    /// update tree info
    rigid_root->updateTree();

    auto idx_map = rigid_root->getJointIdxMap();

    {
        float box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
        float rigid_mass = 12;
        float ixx = 10.0, ixy = 0.0, ixz = 0.0, iyy = 1.0, iyz = 0.0, izz = 10.0;

        Vector3f rigid_r1(0, -box_sy / 2.0, 0);
        Vector3f rigid_r2(0, -box_sy / 2.0, 0);
        Vector3f rigid_r3(0, -box_sy / 2.0, 0);

        /// ********* joint1 info
        Vector3f          joint_r1(0, 9, 0);
        Quaternion<float> joint_q1(Vector3f(1, 1, 1), 1);
        joint1->setJointInfo(-rigid_r1);

        /// ********* joint2 info
        Vector3f          joint_r2(0, -box_sy / 2.0, 0);
        Quaternion<float> joint_q2;
        joint2->setJointInfo(-rigid_r2);

        /// ********* joint1 info
        Vector3f          joint_r3(0, -box_sy / 2.0, 0);
        Quaternion<float> joint_q3;
        joint3->setJointInfo(-rigid_r3);

        /// ********** rigid 1
        int               id1 = rigid1->getId();
        Quaternion<float> rigid_q1;
        motion_state->m_rel_r[id1] = joint_r1 + joint_q1.rotate(rigid_r1);
        motion_state->m_rel_q[id1] = joint_q1 * rigid_q1;
        rigid1->setGeometrySize(box_sx, box_sy, box_sz);
        rigid1->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

        /// ********** rigid 2
        int               id2 = rigid2->getId();
        Quaternion<float> rigid_q2;
        motion_state->m_rel_r[id2] = joint_r2 + joint_q2.rotate(rigid_r2);
        motion_state->m_rel_q[id2] = joint_q2 * rigid_q2;
        rigid2->setGeometrySize(box_sx, box_sy, box_sz);
        rigid2->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

        /// ********** rigid 2
        int               id3 = rigid3->getId();
        Quaternion<float> rigid_q3;
        motion_state->m_rel_r[id3] = joint_r3 + joint_q3.rotate(rigid_r3);
        motion_state->m_rel_q[id3] = joint_q3 * rigid_q3;
        rigid3->setGeometrySize(box_sx, box_sy, box_sz);
        rigid3->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

        std::default_random_engine            e(time(0));
        std::uniform_real_distribution<float> u(-2.0, 2.0);

        /// set velocity
        SpatialVector<float> relv1(u(e), u(e), u(e), 0, 0, 0);                                                   ///< relative velocity, in successor frame.
        joint1->getJointSpace().transposeMul(relv1, &(motion_state->generalVelocity[idx_map[id1]]));             ///< map relative velocity into joint space vector.
        motion_state->m_v[id1] = (joint1->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id1]])));  ///< set

        SpatialVector<float> relv2(u(e), u(e), u(e), 0, 0, 0);                                                   ///< relative velocity, in successor frame.
        joint2->getJointSpace().transposeMul(relv2, &(motion_state->generalVelocity[idx_map[id2]]));             ///< map relative velocity into joint space vector.
        motion_state->m_v[id2] = (joint2->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id2]])));  ///< set

        SpatialVector<float> relv3(u(e), u(e), u(e), 0, 0, 0);                                                   ///< relative velocity, in successor frame.
        joint3->getJointSpace().transposeMul(relv3, &(motion_state->generalVelocity[idx_map[id3]]));             ///< map relative velocity into joint space vector.
        motion_state->m_v[id3] = (joint3->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id3]])));  ///< set

        motion_state->updateGlobalInfo();
    }

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

template <int N = 10>
void demo_MultiRigid()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    //int n = 10;

    RigidBody2_ptr                  rigid[N];
    std::shared_ptr<SphericalJoint> joint[N];

    /// root
    RigidBodyRoot_ptr rigid_root = std::make_shared<RigidBodyRoot<DataType3f>>("rigid_root");
    root->addChild(rigid_root);
    //rigid_root->setGravity(Vector3f(0, 0, 0));

    /// system state
    std::shared_ptr<SystemState>       system_state = rigid_root->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    RigidBody2_ptr last_rigid = rigid_root;
    for (int i = 0; i < N; ++i)
    {
        /// rigid body
        rigid[i] = std::make_shared<RigidBody2<DataType3f>>("rigid");
        last_rigid->addChild(rigid[i]);

        /// joint . rigid[i-1] -> rigid[i]
        joint[i] = std::make_shared<SphericalJoint>("joint");
        joint[i]->setRigidBody(last_rigid.get(), rigid[i].get());
        last_rigid->addChildJoint(joint[i]);
        rigid[i]->setParentJoint(joint[i].get());

        auto renderModule = std::make_shared<RigidMeshRender>(rigid[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        rigid[i]->addVisualModule(renderModule);

        last_rigid = rigid[i];
    }

    /// update tree info
    rigid_root->updateTree();

    auto idx_map = rigid_root->getJointIdxMap();

    std::default_random_engine            e(time(0));
    std::uniform_real_distribution<float> u(-2.0, 2.0);
    std::uniform_real_distribution<float> u1(-1, 1);

    for (int i = 0; i < N; ++i)
    {
        float box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
        box_sx *= 0.1;
        box_sy *= 0.1;
        box_sz *= 0.1;
        float rigid_mass = 12;
        float ixy = 0.0, ixz = 0.0, iyz = 0.0;
        float ixx = (box_sy * box_sy + box_sz * box_sz);
        float iyy = (box_sx * box_sx + box_sz * box_sz);
        float izz = (box_sx * box_sx + box_sy * box_sy);

        Vector3f rigid_r(0, -box_sy / 2.0, 0);

        /// ********* joint info
        Vector3f          joint_r(0, -box_sy / 2.0, 0);
        Quaternion<float> joint_q;  // (Vector3f(u1(e), u1(e), u1(e)), u1(e));
        if (i == 0)
        {
            joint_r[1] = box_sy * (N + 1);
            //joint_q = Quaternion<float>(Vector3f(u1(e), u1(e), u1(e)), u1(e));
        }

        joint[i]->setJointInfo(-rigid_r);

        /// ********** rigid 1
        int               id = rigid[i]->getId();
        Quaternion<float> rigid_q;
        motion_state->m_rel_r[id] = joint_r + joint_q.rotate(rigid_r);
        motion_state->m_rel_q[id] = joint_q * rigid_q;
        rigid[i]->setGeometrySize(box_sx, box_sy, box_sz);
        rigid[i]->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

        if (false)
        {
            /// set velocity
            SpatialVector<float> relv(u(e), u(e), u(e), 0, 0, 0);                                                    ///< relative velocity, in successor frame.
            joint[i]->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[idx_map[id]]));             ///< map relative velocity into joint space vector.
            motion_state->m_v[id] = (joint[i]->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id]])));  ///< set
        }
    }

    motion_state->updateGlobalInfo();

    rigid_root->updateTree();
    system_state->m_activeForce[2] = 10.0;

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

//void demo_HelicalJoint()
//{
//	SceneGraph& scene = SceneGraph::getInstance();
//
//	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
//	root->loadSDF("../Media/bar/bar.sdf", false);
//	root->translate(Vector3f(0.2f, 0.2f, 0));
//
//	// root
//	RigidBodyRoot<DataType3f>* rigid_root = new RigidBodyRoot<DataType3f>("rigid_root");
//	root->addChild(shared_ptr<Node>(rigid_root));
//
//
//	{
//		float box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
//		float rigid_mass = 12;
//		float ixx = 10.0, ixy = 0.0, ixz = 0.0, iyy = 1.0, iyz = 0.0, izz = 10.0;
//
//		// rigid 1
//		RigidBody2<DataType3f>* rigid1 = new RigidBody2<DataType3f>("rigid");
//		GeneralVector<float> rigid_r1(3);
//		rigid_r1[1] = 6;
//		Quaternion<float> rigid_q1; rigid_q1.set(Vector3f(-0.5, -0.0, 0));
//		GeneralMatrix<float> rigid_X1;
//		RigidUtil::getTransformationM(rigid_X1, rigid_q1, rigid_r1);
//		rigid1->setX(rigid_X1);
//		rigid1->setR(rigid_r1);
//		rigid1->setQuaternion(rigid_q1);
//		rigid1->setGeometrySize(2, 2, 2);
//
//		rigid1->setMass(rigid_mass);
//		GeneralMatrix<float> cur_i(6, 6);
//		cur_i(0, 0) = 8;	cur_i(0, 1) = ixy;	cur_i(0, 2) = ixz;
//		cur_i(1, 0) = ixy;	cur_i(1, 1) = 8;	cur_i(1, 2) = iyz;
//		cur_i(2, 0) = ixz;	cur_i(2, 1) = iyz;	cur_i(2, 2) = 8;
//		cur_i(3, 3) = rigid_mass;
//		cur_i(4, 4) = rigid_mass;
//		cur_i(5, 5) = rigid_mass;
//		rigid1->setI(cur_i);
//
//
//
//		// rigid 2
//		RigidBody2<DataType3f>* rigid2 = new RigidBody2<DataType3f>("rigid");
//		GeneralVector<float> rigid_r2(3);
//		Quaternion<float> rigid_q2;
//		GeneralMatrix<float> rigid_X2;
//		RigidUtil::getTransformationM(rigid_X2, rigid_q2, rigid_r2);
//		rigid2->setX(rigid_X2);
//		rigid2->setR(rigid_r2);
//		rigid2->setQuaternion(rigid_q2);
//		rigid2->setGeometrySize(1, 7, 1);
//
//		rigid2->setMass(rigid_mass);
//		//GeneralMatrix<float> cur_i(6, 6);
//		cur_i(0, 0) = 50;	cur_i(0, 1) = ixy;	cur_i(0, 2) = ixz;
//		cur_i(1, 0) = ixy;	cur_i(1, 1) = 2;	cur_i(1, 2) = iyz;
//		cur_i(2, 0) = ixz;	cur_i(2, 1) = iyz;	cur_i(2, 2) = 50;
//		cur_i(3, 3) = rigid_mass;
//		cur_i(4, 4) = rigid_mass;
//		cur_i(5, 5) = rigid_mass;
//		rigid2->setI(cur_i);
//
//
//		// joint 1
//		std::shared_ptr<FixedJoint> base_joint1(new FixedJoint("from_base"));
//		base_joint1->setRigidBody(rigid_root, rigid1);
//		rigid_root->addChildJoint(base_joint1);
//		rigid1->setParentJoint(base_joint1);
//		rigid_root->addChild(shared_ptr<RigidBody2<DataType3f>>(rigid1));
//
//
//		// joint 2
//		std::shared_ptr<HelicalJoint> helical_joint(new HelicalJoint("planar_joint"));
//		helical_joint->setRigidBody(rigid1, rigid2);
//		rigid1->addChildJoint(helical_joint);
//		rigid2->setParentJoint(helical_joint);
//		rigid1->addChild(shared_ptr<RigidBody2<DataType3f>>(rigid2));
//
//		GeneralVector<float> joint_r2(3);	//joint_r2[1] = 1.0;
//		Quaternion<float> joint_q2;
//		GeneralMatrix<float> joint_X2;
//		RigidUtil::getTransformationM(joint_X2, joint_q2, joint_r2);
//		helical_joint->setXT(joint_X2);
//		helical_joint->setR(joint_r2);
//		helical_joint->setQuaternion(joint_q2);
//
//		GeneralVector<float> axis_norm(3);
//		axis_norm[1] = 1.0;
//		helical_joint->setJointInfo(axis_norm, 0.2);
//
//	}
//
//	rigid_root->initTreeGlobalInfo();
//
//	GLApp window;
//	window.createWindow(1024, 768);
//	window.mainLoop();
//
//}
//
//
//void demo_CylindricalJoint()
//{
//	SceneGraph& scene = SceneGraph::getInstance();
//
//	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
//	root->loadSDF("../Media/bar/bar.sdf", false);
//	root->translate(Vector3f(0.2f, 0.2f, 0));
//
//	// root
//	RigidBodyRoot<DataType3f>* rigid_root = new RigidBodyRoot<DataType3f>("rigid_root");
//	root->addChild(shared_ptr<Node>(rigid_root));
//
//
//	{
//		float box_sx = 1.0, box_sy = 3.0, box_sz = 1.0;
//		float rigid_mass = 12;
//		float ixx = 10.0, ixy = 0.0, ixz = 0.0, iyy = 1.0, iyz = 0.0, izz = 10.0;
//
//		// rigid 1
//		RigidBody2<DataType3f>* rigid1 = new RigidBody2<DataType3f>("rigid");
//		GeneralVector<float> rigid_r1(3);
//		rigid_r1[1] = 6;
//		Quaternion<float> rigid_q1; //rigid_q1.set(Vector3f(-0.5, -0.0, 0));
//		GeneralMatrix<float> rigid_X1;
//		RigidUtil::getTransformationM(rigid_X1, rigid_q1, rigid_r1);
//		rigid1->setX(rigid_X1);
//		rigid1->setR(rigid_r1);
//		rigid1->setQuaternion(rigid_q1);
//		rigid1->setGeometrySize(2, 2, 2);
//
//		rigid1->setMass(rigid_mass);
//		GeneralMatrix<float> cur_i(6, 6);
//		cur_i(0, 0) = 8;	cur_i(0, 1) = ixy;	cur_i(0, 2) = ixz;
//		cur_i(1, 0) = ixy;	cur_i(1, 1) = 8;	cur_i(1, 2) = iyz;
//		cur_i(2, 0) = ixz;	cur_i(2, 1) = iyz;	cur_i(2, 2) = 8;
//		cur_i(3, 3) = rigid_mass;
//		cur_i(4, 4) = rigid_mass;
//		cur_i(5, 5) = rigid_mass;
//		rigid1->setI(cur_i);
//
//
//
//		// rigid 2
//		RigidBody2<DataType3f>* rigid2 = new RigidBody2<DataType3f>("rigid");
//		GeneralVector<float> rigid_r2(3);
//		Quaternion<float> rigid_q2;
//		GeneralMatrix<float> rigid_X2;
//		RigidUtil::getTransformationM(rigid_X2, rigid_q2, rigid_r2);
//		rigid2->setX(rigid_X2);
//		rigid2->setR(rigid_r2);
//		rigid2->setQuaternion(rigid_q2);
//		rigid2->setGeometrySize(1, 7, 1);
//
//		rigid2->setMass(rigid_mass);
//		//GeneralMatrix<float> cur_i(6, 6);
//		cur_i(0, 0) = 50;	cur_i(0, 1) = ixy;	cur_i(0, 2) = ixz;
//		cur_i(1, 0) = ixy;	cur_i(1, 1) = 2;	cur_i(1, 2) = iyz;
//		cur_i(2, 0) = ixz;	cur_i(2, 1) = iyz;	cur_i(2, 2) = 50;
//		cur_i(3, 3) = rigid_mass;
//		cur_i(4, 4) = rigid_mass;
//		cur_i(5, 5) = rigid_mass;
//		rigid2->setI(cur_i);
//
//
//		// joint 1
//		std::shared_ptr<FixedJoint> base_joint1(new FixedJoint("from_base"));
//		base_joint1->setRigidBody(rigid_root, rigid1);
//		rigid_root->addChildJoint(base_joint1);
//		rigid1->setParentJoint(base_joint1);
//		rigid_root->addChild(shared_ptr<RigidBody2<DataType3f>>(rigid1));
//
//
//		// joint 2
//		std::shared_ptr<CylindricalJoint> cylindrical_joint(new CylindricalJoint("planar_joint"));
//		cylindrical_joint->setRigidBody(rigid1, rigid2);
//		rigid1->addChildJoint(cylindrical_joint);
//		rigid2->setParentJoint(cylindrical_joint);
//		rigid1->addChild(shared_ptr<RigidBody2<DataType3f>>(rigid2));
//
//		GeneralVector<float> joint_r2(3);	//joint_r2[1] = 1.0;
//		Quaternion<float> joint_q2;
//		GeneralMatrix<float> joint_X2;
//		RigidUtil::getTransformationM(joint_X2, joint_q2, joint_r2);
//		cylindrical_joint->setXT(joint_X2);
//		cylindrical_joint->setR(joint_r2);
//		cylindrical_joint->setQuaternion(joint_q2);
//
//		GeneralVector<float> axis_norm(3);
//		axis_norm[1] = 1.0;
//		cylindrical_joint->setJointInfo(axis_norm);
//
//	}
//
//	rigid_root->initTreeGlobalInfo();
//
//	GLApp window;
//	window.createWindow(1024, 768);
//	window.mainLoop();
//
//}