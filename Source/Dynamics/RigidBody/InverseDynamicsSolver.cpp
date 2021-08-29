
#include "Dynamics/RigidBody/InverseDynamicsSolver.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Framework/Action/Action.h"
#include <queue>
#include <memory>
#include <vector>

#include "Transform3d.h"
#include "RigidBodyRoot.h"
#include "SystemState.h"

namespace PhysIKA {

InverseDynamicsSolver::InverseDynamicsSolver(Node* parent_node)
    : m_parent_node(parent_node)
{
}

void InverseDynamicsSolver::inverseDynamics(const SystemState& system_state, const Vectornd<float>& ddq, Vectornd<float>& tau, bool zeroAcceleration)
{
    Vectornd<SpatialVector<float>> avp;
    Vectornd<SpatialVector<float>> v;
    //Vectornd<Transform3d<float>> Xup;
    Vectornd<SpatialVector<float>> fvp;

    //std::vector< std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>> all_node =  static_cast<RigidBody2<DataType3f>*>(this->getParent())->getAllParentidNodePair();

    const SystemMotionState& s = *(system_state.m_motionState);

    RigidBodyRoot<DataType3f>*                                           root     = static_cast<RigidBodyRoot<DataType3f>*>(this->m_parent_node);
    std::vector<std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>> all_node = root->getAllParentidNodePair();
    const std::vector<int>&                                              idx_map  = root->getJointIdxMap();

    for (int i = 0; i < all_node.size(); ++i)
    {
        int                                     parent_id = all_node[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_node[i].second;
        Joint*                                  cur_joint = cur_node->getParentJoint();
        std::shared_ptr<RigidBody2<DataType3f>> parent_node(static_cast<RigidBody2<DataType3f>*>(cur_joint->getParent()));

        {
            //MatrixMN<float> XJ = cur_joint->getXJ();
            //MatrixMN<float> XT = cur_joint->getXT();
            //Xup.push_back(XJ*XT);

            //Xup[i] = cur_node->getX();

            //MatrixMN<float> Si = cur_joint->getS();
            const JointSpaceBase<float>& Si = cur_joint->getJointSpace();

            SpatialVector<float> vJ = Si.mul(&(s.generalVelocity[idx_map[i]]));

            if (parent_id < 0)
            {
                v[i] = vJ;

                avp[i] = Si.mul(&(ddq[idx_map[i]]));
            }
            else
            {
                v[i]   = (s.m_X[i].transformM(v[parent_id]) + vJ);
                avp[i] = s.m_X[i].transformM(avp[parent_id]) + Si.mul(&(ddq[idx_map[i]])) + v[i].crossM(vJ);
            }
            fvp[i] = cur_node->getI() * avp[i] + v[i].crossF(cur_node->getI() * v[i]);
            fvp[i] = fvp[i] - system_state.m_externalForce[i];
        }
    }

    //std::vector<Vectornd<float>> tau(all_node.size());
    for (int i = all_node.size() - 1; i >= 0; --i)
    {
        int                                     parent_id = all_node[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_node[i].second;
        Joint*                                  cur_joint = cur_node->getParentJoint();

        //tau[i] = cur_joint->getS().transpose() * fvp[i];
        cur_joint->getJointSpace().transposeMul(fvp[i], &(tau[idx_map[i]]));

        if (parent_id >= 0)
        {
            fvp[parent_id] = fvp[parent_id] + s.m_X[i].inverseTransform().transformF(fvp[i]);
        }
    }

    //return tau;
}

}  // namespace PhysIKA