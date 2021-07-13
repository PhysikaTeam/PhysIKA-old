#include "Dynamics/RigidBody/FeatherstoneIntegrationModule.h"

#include "Dynamics/RigidBody/RigidBodyRoot.h"
#include "Dynamics/RigidBody/ArticulatedBodyFDSolver.h"

PhysIKA::FeatherstoneIntegrationModule::FeatherstoneIntegrationModule()
{
    m_fd_solver = std::make_shared<ArticulatedBodyFDSolver>();
    m_fd_solver->setParent(this->getParent());
}

void PhysIKA::FeatherstoneIntegrationModule::begin()
{
    if (!m_time_init)
    {
        //m_dt = 0.001;
        m_dt        = 1.0 / 60;
        m_last_time = clock() / 1000.0;
        m_time_init = true;
    }
    else
    {
        double cur_time = clock() / 1000.0;
        m_dt            = cur_time - m_last_time;
        m_last_time     = cur_time;
    }
}

bool PhysIKA::FeatherstoneIntegrationModule::execute()
{
    RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(this->getParent());
    SystemState&               s    = *(root->getSystemState());

    int ndof = root->getJointDof();
    m_ddq.resize(ndof);

    // Solve forward dynamics
    bool solve_ok = m_fd_solver->solve(s, *(s.m_motionState), m_ddq);

    if (!solve_ok)
    {
        return false;
    }

    // Joint index map.
    const std::vector<int>& idx_map = root->getJointIdxMap();

    const std::vector<std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>>& all_nodes = root->getAllParentidNodePair();
    Vectornd<float>&                                                            statedq   = s.m_motionState->generalVelocity;
    Vectornd<float>&                                                            stateq    = s.m_motionState->generalPosition;
    //std::vector<SpatialVector<float>>& stateRelV = s.m_motionState->m_v;
    std::vector<SpatialVector<float>>& gloVel = s.m_motionState->getGlobalVelocity();
    std::vector<Vector3f>&             gloPos = s.m_motionState->getGlobalPosition();
    std::vector<Quaternion<float>>&    gloRot = s.m_motionState->getGlobalRotation();

    // Update general position.
    for (int i = 0; i < ndof; ++i)
    {
        stateq[i] += statedq[i] * m_dt;
        statedq[i] += m_ddq[i] * m_dt;
    }

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        int    parentid = all_nodes[i].first;
        auto   prigid   = all_nodes[i].second;
        auto   pjoint   = prigid->getParentJoint();
        float* curdq    = &(statedq[idx_map[i]]);
        float* curq     = &(stateq[idx_map[i]]);

        // Transform from successor to predecessor.
        Transform3d<float> trans;
        pjoint->getRelativeTransform(curq, trans);
        trans = pjoint->getXT() * trans;

        // Get rotation and translation information from transform object.
        gloRot[i] = trans.getRotation();
        gloPos[i] = trans.getRotation().rotate(-trans.getTranslation());
        if (parentid >= 0)
        {
            gloPos[i] = gloPos[parentid] + gloRot[parentid].rotate(gloPos[i]);
            gloRot[i] = gloRot[parentid] * gloRot[i];
        }

        // GLobal velocity.
        gloVel[i] = pjoint->getJointSpace().mul(curdq);
        if (parentid >= 0)
        {
            gloVel[i] += gloVel[parentid];
        }
    }

    // update global velocity and position
    //?????

    return true;
}
