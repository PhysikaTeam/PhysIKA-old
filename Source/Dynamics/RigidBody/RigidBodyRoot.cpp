#include "RigidBodyRoot.h"
#include "Dynamics/RigidBody/RigidTimeIntegrationModule.h"
//#include "Dynamics/RigidBody/RigidUtil.h"

#include "Core/Utility/CTimer.h"
#include <iostream>

#include "Core/Vector/vector_nd.h"

#include "RigidDebugInfoModule.h"
#include "Dynamics/RigidBody/RigidUtil.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(RigidBodyRoot, TDataType)

template <typename TDataType>
PhysIKA::RigidBodyRoot<TDataType>::RigidBodyRoot(std::string name)
    : RigidBody2(name)  //, m_state()
{
    this->initSystemState();
    //m_state->m_motionState = std::make_shared<SystemMotionState>(this);
}

template <typename TDataType>
RigidBodyRoot<TDataType>::~RigidBodyRoot()
{
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::advance(Real dt)
{
    //return;

    CTimer timer1;
    timer1.start();

    updateTree();

    timer1.stop();
    //std::cout << "TIME * updateTree:  " << timer1.getElapsedTime() << std::endl;

    CTimer timer2;
    timer2.start();

    // Prepare system force state.
    this->collectForceState();
    //this->_collectMotionState();
    this->_collectRelativeMotionState();

    // update simulation variables
    auto time_integrator = this->getModule<RigidTimeIntegrationModule>();

    time_integrator->setDt(dt);
    //time_integrator->set
    time_integrator->begin();
    //time_integrator->setDt(dt);
    time_integrator->execute();
    time_integrator->end();

    timer2.stop();
    //std::cout << ":TIME * execute:  " << timer2.getElapsedTime() << std::endl;

    this->_updateTreeGlobalInfo();
    this->_clearRIgidForce();

#ifdef _DEBUG
    auto debugInfo = this->getModule<RigidDebugInfoModule>();
    debugInfo->begin();
    //debugInfo->execute();
    debugInfo->end();

#endif  // _DEBUG

    //CTimer timer3;
    //timer3.start();

    ////Vectornd<float> mom = this->calMomentum6();
    ////float eng = this->calKineticEnergy();
    //Vectornd<float> mom;
    //float eng;
    //this->calEnergyAndMomentum(mom, eng);

    //std::cout << "E:  " << eng << std::endl;
    //std::cout << "MOM: "; mom.out();
    ////std::cout << std::endl;

    //timer3.stop();
    //std::cout << ":TIME * cal Energy: " << timer3.getElapsedTime() << std::endl;
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::updateTopology()
{
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::addLoopJoint(Joint* joint)
{
    m_loopJoints.push_back(std::make_shared<Joint>(*joint));
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::addLoopJoint(std::shared_ptr<Joint> joint)
{
    m_loopJoints.push_back(joint);
}

template <typename TDataType>
bool RigidBodyRoot<TDataType>::initialize()
{
    //this->_updateTreeGlobalInfo();
    this->setVisible(false);

    std::shared_ptr<RigidTimeIntegrationModule> time_integrator(new RigidTimeIntegrationModule());
    this->addModule<RigidTimeIntegrationModule>(time_integrator);

#ifdef _DEBUG
    std::shared_ptr<RigidDebugInfoModule> debugInfoModule = std::make_shared<RigidDebugInfoModule>();
    this->addModule<RigidDebugInfoModule>(debugInfoModule);
#endif  // _DEBUG

    this->updateTree();

    this->_collectRelativeMotionState();
    //this->_collectMotionState();
    this->_updateTreeGlobalInfo();

    return true;
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::_updateTreeGlobalInfo()
{
    //std::vector< std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>>& all_nodes = this->m_all_childs_node_pairs;
    std::vector<std::shared_ptr<RigidBody2<DataType3f>>>& all_nodes = this->m_all_childs_nodes;

    SystemMotionState& s = *(this->m_state->m_motionState);

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        std::shared_ptr<RigidBody2<DataType3f>> cur_node = all_nodes[i];

        Transform3d<float>   movetobody(s.globalPosition[i], Quaternion<float>(0, 0, 0, 1));
        SpatialVector<float> glov = movetobody.transformM(s.globalVelocity[i]);
        cur_node->setLinearVelocity(Vector3f(glov[3], glov[4], glov[5]));
        cur_node->setAngularVelocity(Vector3f(glov[0], glov[1], glov[1]));

        cur_node->setGlobalR(s.globalPosition[i]);
        cur_node->setGlobalQ(s.globalRotation[i]);

        cur_node->setRelativeQ(s.m_rel_q[i]);
        cur_node->setRelativeR(s.m_rel_r[i]);

        cur_node->updateTopology();
    }
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::_collectMotionState()
{
    std::vector<std::shared_ptr<RigidBody2<DataType3f>>>&                all_nodes  = this->m_all_childs_nodes;
    std::vector<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& node_pairs = m_all_childs_node_pairs;
    SystemMotionState&                                                   s          = *(this->m_state->m_motionState);
    s.setNum(all_nodes.size(), this->getJointDof());

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        int                                     parentid = node_pairs[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node = all_nodes[i];
        auto                                    pjoint   = cur_node->getParentJoint();
        //auto S = cur_node->getParentJoint()->getJointSpace();
        int curdof = pjoint->getJointDOF();
        int idx0   = m_joint_idx_map[i];

        s.globalPosition[i] = cur_node->getGlobalR();
        s.globalRotation[i] = cur_node->getGlobalQ();

        if (parentid >= 0)
        {
            s.m_rel_q[i] = s.globalRotation[parentid].getConjugate() * s.globalRotation[i];
            s.m_rel_r[i] = s.globalRotation[parentid].getConjugate().rotate(s.globalPosition[i] - s.globalPosition[parentid]);
        }
        else
        {
            s.m_rel_q[i] = s.globalRotation[i];
            s.m_rel_r[i] = s.globalPosition[i];
        }
        s.m_X[i] = Transform3d<float>(s.m_rel_r[i], s.m_rel_q[i].getConjugate());

        //s.globalVelocity[i] = SpatialVector<float> (cur_node->getAngularVelocity(), cur_node->getLinearVelocity());
        //SpatialVector<float> relv = s.globalVelocity[i] - s.globalVelocity[parentid];
        //
        //RigidUtil::setStxS(S.getBases(), curdof,
        //    s.m_rel_q[i].getConjugate().rotate(s.globalVelocity[i] - s.globalVelocity[parentid]),
        //    )
        //s.m_v[i] = RigidUtil::setSxq(S.getBases(), )

        //

        //S.transposeMul(s.m_v[i],
        //    &(s.generalVelocity[m_joint_idx_map[i]]));

        for (int j = 0; j < curdof; ++j)
        {
            s.generalPosition[idx0 + j] = pjoint->getInitGeneralPos()[j];
        }
    }
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::_collectRelativeMotionState()
{
    std::vector<std::shared_ptr<RigidBody2<DataType3f>>>&                all_nodes  = this->m_all_childs_nodes;
    std::vector<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& node_pairs = m_all_childs_node_pairs;
    SystemMotionState&                                                   s          = *(this->m_state->m_motionState);
    s.setNum(all_nodes.size(), this->getJointDof());

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        int                                     parentid = node_pairs[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node = all_nodes[i];
        auto                                    pjoint   = cur_node->getParentJoint();
        //auto S = cur_node->getParentJoint()->getJointSpace();
        int curdof = pjoint->getJointDOF();
        int idx0   = m_joint_idx_map[i];

        s.m_rel_q[i] = cur_node->getRelativeQ();
        s.m_rel_r[i] = cur_node->getRelativeR();

        if (parentid >= 0)
        {
            s.globalRotation[i] = s.globalRotation[parentid] * s.m_rel_q[i];
            s.globalPosition[i] = s.globalPosition[parentid] + s.globalRotation[parentid].rotate(s.m_rel_r[i]);

            //s.m_rel_q[i] = s.globalRotation[parentid].getConjugate() * s.globalRotation[i];
            //s.m_rel_r[i] = s.globalRotation[parentid].getConjugate().rotate(s.globalPosition[i] - s.globalPosition[parentid]);
        }
        else
        {
            s.globalRotation[i] = s.m_rel_q[i];
            s.globalPosition[i] = s.m_rel_r[i];
            //s.m_rel_q[i] = s.globalRotation[i];
            //s.m_rel_r[i] = s.globalPosition[i];
        }
        s.m_X[i] = Transform3d<float>(s.m_rel_r[i], s.m_rel_q[i].getConjugate());

        for (int j = 0; j < curdof; ++j)
        {
            s.generalPosition[idx0 + j] = pjoint->getInitGeneralPos()[j];
        }
    }
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::collectForceState()
{
    std::vector<std::shared_ptr<RigidBody2<DataType3f>>>&                all_nodes  = this->m_all_childs_nodes;
    std::vector<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& node_pairs = m_all_childs_node_pairs;
    SystemMotionState&                                                   s          = *(this->m_state->m_motionState);
    s.setNum(all_nodes.size(), this->getJointDof());
    std::vector<SpatialVector<float>>& externalForce = m_state->m_externalForce;
    Vectornd<float>&                   activeForce   = m_state->m_activeForce;

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        int                                     parentid = node_pairs[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> prigid   = all_nodes[i];
        auto                                    pjoint   = prigid->getParentJoint();
        int                                     curdof   = pjoint->getJointDOF();
        int                                     idx0     = m_joint_idx_map[i];

        // External force and torque that applied to rigid body.
        externalForce[i] = SpatialVector<float>(prigid->getExternalTorque(),
                                                prigid->getExternalForce());

        // Joint active force. (Motor force)
        const float* curMotor = pjoint->getMotorForce();
        for (int di = 0; di < curdof; ++di)
        {
            activeForce[idx0 + di] = curMotor[di];
        }
    }
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::_clearRIgidForce()
{
    std::vector<std::shared_ptr<RigidBody2<DataType3f>>>& all_nodes = this->m_all_childs_nodes;

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        std::shared_ptr<RigidBody2<DataType3f>> prigid = all_nodes[i];

        prigid->setExternalForce(Vector3f(0, 0, 0));
        prigid->setExternalTorque(Vector3f(0, 0, 0));
    }
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::updateTree()
{

    m_all_childs_nodes.clear();
    m_all_childs_node_pairs.clear();
    Node* root = this;

    // pair: first->parent id;  second->node pointer;
    std::queue<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>> que;

    ListPtr<Node> root_childs = root->getChildren();
    //int cur_id = 0;
    for (auto iter = root_childs.begin(); iter != root_childs.end(); ++iter)
    {
        que.push(std::make_pair(-1, std::dynamic_pointer_cast<RigidBody2<TDataType>>(*iter)));
    }

    int i = -1;
    while (!que.empty())
    {
        ++i;

        std::shared_ptr<RigidBody2<TDataType>> cur_node = que.front().second;
        m_all_childs_node_pairs.push_back(que.front());
        m_all_childs_nodes.push_back(cur_node);
        que.pop();

        cur_node->setId(i);

        ListPtr<Node> node_childs = cur_node->getChildren();
        for (auto iter = node_childs.begin(); iter != node_childs.end(); ++iter)
        {
            que.push(std::make_pair(i, std::dynamic_pointer_cast<RigidBody2<TDataType>>(*iter)));
        }
    }

    _calculateDof();

    if (this->m_state)
    {
        this->m_state->setNum(this->m_all_childs_nodes.size(), this->m_joint_dof);
    }

    //m_state.build();
}

template <typename TDataType>
void RigidBodyRoot<TDataType>::initSystemState()
{

    this->m_state                = std::make_shared<SystemState>(this);
    this->m_state->m_motionState = std::make_shared<SystemMotionState>(this);

    this->m_state->m_gravity = Vector3f(0, -9, 0);
}

//template<typename TDataType>
//void RigidBodyRoot<TDataType>::initTreeGlobalInfo()
//{
//    this->updateTree();

//    for (int i = 0; i < m_all_childs_nodes.size(); ++i)
//    {
//        std::shared_ptr<RigidBody2<TDataType>> cur_rigid = m_all_childs_nodes[i];
//        std::shared_ptr<Joint> parent_joint = cur_rigid->getParentJoint();
//        RigidBody2<TDataType>* parent_rigid = static_cast<RigidBody2<TDataType>*>(parent_joint->getParent());

//        cur_rigid->setX(cur_rigid->getX() * parent_joint->getXT());

//        Vector3f rotated_r = parent_joint->getQuaternion().rotate(RigidUtil::toVector3f(cur_rigid->getR()));
//        cur_rigid->setR(RigidUtil::toVectornd(rotated_r) + parent_joint->getR());
//        cur_rigid->setQuaternion(cur_rigid->getQuaternion() * parent_joint->getQuaternion());

//        Vectornd<float> cur_r = cur_rigid->getR();
//        Quaternion<float> cur_q = cur_rigid->getQuaternion();
//        cur_rigid->setGlobalR(parent_rigid->getGlobalQ().rotate(Vector3f(cur_r(0), cur_r(1), cur_r(2))) + parent_rigid->getGlobalR());
//        cur_rigid->setGlobalQ(parent_rigid->getGlobalQ() * cur_q);

//    }
//}

template <typename TDataType>
int RigidBodyRoot<TDataType>::_calculateDof()
{
    int dof = 0;
    m_joint_idx_map.resize(m_all_childs_nodes.size(), 0);

    if (m_all_childs_nodes.size() > 0)
    {
        dof += (m_all_childs_nodes[0])->getParentJoint()->getJointDOF();
    }

    for (int i = 1; i < m_all_childs_nodes.size(); ++i)
    {
        int cur_dof = (m_all_childs_nodes[i])->getParentJoint()->getJointDOF();
        dof += cur_dof;
        m_joint_idx_map[i] = (m_all_childs_nodes[i - 1])->getParentJoint()->getJointDOF() + m_joint_idx_map[i - 1];
    }
    m_joint_dof = dof;
    return dof;
}

//template<typename TDataType>
//void RigidBodyRoot<TDataType>::_getAccelerationQ(std::vector<Vectornd<float>>& ddq)
//{
//    std::vector<std::shared_ptr<RigidBody2<TDataType>>> all_nodes = getAllNode();

//    ddq.resize(all_nodes.size());

//    for (int i = 0; i < all_nodes.size(); ++i)
//    {
//        auto parent_joint = (all_nodes[i])->getParentJoint();
//        ddq[i] = parent_joint->getddq();
//    }
//}

}  // namespace PhysIKA