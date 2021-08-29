#include "RigidDebugInfoModule.h"
#include "RigidBodyRoot.h"

#include <iostream>

namespace PhysIKA {
IMPLEMENT_CLASS(RigidDebugInfoModule)

bool RigidDebugInfoModule::execute()
{
    RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(this->getParent());

    const std::vector<std::pair<int, RigidBody2_ptr>>& all_nodes = root->getAllParentidNodePair();
    SystemMotionState&                                 state     = *(root->getSystemState()->m_motionState);

    float                             eng = 0.0;
    SpatialVector<float>              mom;
    std::vector<SpatialVector<float>> all_v(all_nodes.size());

    for (int i = 0; i < all_nodes.size(); ++i)
    {
        int            parent_id = all_nodes[i].first;
        RigidBody2_ptr cur_node  = all_nodes[i].second;
        Joint*         cur_joint = cur_node->getParentJoint();

        /// Transformation from world to node.
        Transform3d<float> Xo(state.globalPosition[i] - Vector3f(0, 12, 0), state.globalRotation[i].getConjugate());

        /// Transformation from predecessor to successor.
        Transform3d<float>& Xup = state.m_X[i];

        /// global velocity in node frame.
        if (parent_id >= 0)
        {
            all_v[i] = state.m_v[i] + Xup.transformM(all_v[parent_id]);
        }
        else
        {
            all_v[i] = state.m_v[i];
        }

        /// Momemtum of current node, expressed in world frame.
        SpatialVector<float> cur_mom;
        cur_mom = Xo.inverseTransform().transformF(cur_node->getI() * all_v[i]);

        /// Kinetic energy of current node.
        float cur_kin;
        cur_kin = Xo.inverseTransform().transformM(all_v[i]) * cur_mom * 0.5;

        /// Potential energy of current node.
        float cur_pot;
        cur_pot = root->getGravity().dot(state.globalPosition[i]) * (cur_node->getI().getMass() * (-1));

        /// Total momemtum
        mom = mom + cur_mom;

        /// Total energy
        eng = eng + cur_kin + cur_pot;
    }

    //std::cout << "[**] dq";
    //for (int i = 0; i < state.m_dq.size(); ++i)
    //{
    //    std::cout << "    " << state.m_dq[i];
    //}
    //std::cout << std::endl;
    std::cout << "MOMEMTUM:  " << mom[0] << "  " << mom[1] << "  " << mom[2] << "  " << mom[3] << "  " << mom[4] << "  " << mom[5] << std::endl;
    std::cout << "ENERGY:    " << eng << std::endl;
    return true;
}

}  // namespace PhysIKA