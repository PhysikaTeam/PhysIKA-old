#pragma once

#include "Dynamics/RigidBody/SystemMotionState.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "ForwardDynamicsSolver.h"

namespace PhysIKA {
//class SemiImplicitEulerIntegrator
//{
//public:

//    void solve(SystemMotionState& s0, double dt)
//    {
//        RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(s0.m_root);

//        MatrixMN<float>& H = root->getH();
//        Vectornd<float>& C = root->getC();

//        // build motion equation
//        InertiaMatrixFDSolver forward_dynamics;
//        forward_dynamics.setParent(root);
//        forward_dynamics.buildJointSpaceMotionEquation<RigidState>(s0, H, C, Vectornd<float>(root->getJointDof())); ///?????????????????????

//        Vectornd<float> ddq;
//        if (H.linearSolve(C.negative(), ddq))
//        {

//            //RigidState s0(root);
//            std::vector< std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>>& all_nodes = root->getAllParentidNodePair();

//            //s0.m_dq.resize(all_nodes.size());
//            //s0.m_r.resize(all_nodes.size());
//            //s0.m_qua.resize(all_nodes.size());
//            //s0.m_v.resize(all_nodes.size());

//            //s0.m_X.resize(all_nodes.size());
//            //s0.m_global_r.resize(all_nodes.size());
//            //s0.m_global_q.resize(all_nodes.size());

//            // build index map
//            std::vector<int>& idx_map = root->getJointIdxMap();

//            for (int i = 0; i < all_nodes.size(); ++i)
//            {
//                int parent_id = all_nodes[i].first;
//                std::shared_ptr<RigidBody2<DataType3f>> cur_node = all_nodes[i].second;
//                std::shared_ptr<Joint> cur_joint = cur_node->getParentJoint();

//                int dof = cur_joint->getJointDOF();
//                if (dof > 0)
//                {
//                    MatrixMN<float> Xup = s0.m_X[i];
//                    MatrixMN<float> Xupinv = RigidUtil::transformationF2M(Xup.transpose());

//                    // d_dq
//                    Vectornd<float> dq(dof);
//                    const Vectornd<float>& dq0 = s0.getDq(i);
//                    for (int j = 0; j < dof; ++j)
//                    {
//                        dq[j] = dq0[j] + ddq(idx_map[i] + j) * dt;
//                    }
//                    s0.m_dq[i] = dq;

//                    // vJ
//                    s0.m_v[i] = cur_joint->getS() * dq;

//                    Vectornd<float> cur_v6 = s0.m_v[i];
//                    s0.m_qua[i] = s0.m_qua[i] + s0.m_qua[i] * Quaternion<float>(cur_v6[0], cur_v6[1], cur_v6[2], 0) * 0.5 * dt;
//                    cur_v6 = Xupinv * cur_v6;
//                    Vectornd<float> cur_w(3);    cur_w[0] = cur_v6[0];    cur_w[1] = cur_v6[1];    cur_w[2] = cur_v6[2];
//                    Vectornd<float> d_r(3); d_r[0] = cur_v6[3]; d_r[1] = cur_v6[4]; d_r[2] = cur_v6[5];
//                    s0.m_r[i] = s0.m_r[i] + (d_r + cur_w.cross(s0.m_r[i]))* dt;

//                    RigidUtil::getTransformationM(s0.m_X[i], s0.m_qua[i], s0.m_r[i]);
//                    if (parent_id >= 0)
//                    {
//                        s0.m_global_r[i] = s0.m_global_q[parent_id].rotate(RigidUtil::toVector3f(s0.m_r[i])) + s0.m_global_r[parent_id];
//                        s0.m_global_q[i] = (s0.m_global_q[parent_id] * s0.m_qua[i]).normalize();
//                    }
//                    else
//                    {
//                        s0.m_global_r[i] = RigidUtil::toVector3f(s0.m_r[i]);
//                        s0.m_global_q[i] = s0.m_qua[i].normalize();
//                    }
//                }
//                else
//                {
//                    s0.m_dq[i] = Vectornd<float>();

//                    // d_vJ
//                    s0.m_v[i] = Vectornd<float>(6);                    // d_vJ, expressed in node frame

//                                                                            // d_r, d_quaternion
//                    s0.m_qua[i] = Quaternion<float>(0, 0, 0, 0);
//                    s0.m_r[i] = Vectornd<float>(3);
//                }
//            }
//        }
//    }

//};
}  // namespace PhysIKA
