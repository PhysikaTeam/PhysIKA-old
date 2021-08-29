#include "Dynamics/RigidBody/RigidState.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"

namespace PhysIKA {
//    RigidState RigidState::operator*(float t) const
//    {
//        RigidState s(this->m_root);
//
//        s.m_dq = _vecMul(m_dq, t);
//        s.m_r = _vecMul(m_r, t);
//        s.m_qua = _vecMul(m_qua, t);
//        s.m_v = _vecMul(m_v, t);
//
//        return s;
//    }
//
//    RigidState RigidState::operator+(const RigidState & state) const
//    {
//        //const RigidState& state = static_cast<const RigidState&>(state);
//        RigidState s(this->m_root);
//
//        s.m_dq = _vecAdd(m_dq, state.m_dq);
//        s.m_r = _vecAdd(m_r, state.m_r);
//        s.m_qua = _vecAdd(m_qua, state.m_qua);
//        s.m_v = _vecAdd(m_v, state.m_v);
//        //s.m_v = _vecMul(s.m_v, 0.999);            // dissipation
//
//        //update
//        int n = m_dq.size();
//
//        s.m_X.resize(n);
//        for (int i = 0; i < n; ++i)
//        {
//            RigidUtil::getTransformationM(s.m_X[i], s.m_qua[i], s.m_r[i]);
//        }
//
//        RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(m_root);
//
//        auto & all_nodes = root->getAllParentidNodePair();
//
//        s.m_global_r.resize(n);
//        for (int i = 0; i < n; ++i)
//        {
//            int parent_id = all_nodes[i].first;
//            //s.m_global_r[i] = this->m_global_q[(all_nodes[i]).first].rotate(RigidUtil::toVector3f(s.m_r[i]));
//            if (parent_id >= 0)
//            {
//                s.m_global_r[i] = this->m_global_q[parent_id].rotate(RigidUtil::toVector3f(s.m_r[i])) + this->m_global_r[parent_id]; //!!!! 有问题
//            }
//            else
//            {
//                s.m_global_r[i] = RigidUtil::toVector3f(s.m_r[i]);
//            }
//        }
//
//        s.m_global_q.resize(n);
//        for (int i = 0; i < n; ++i)
//        {
//            int parent_id = all_nodes[i].first;
//            if (parent_id >= 0)
//            {
//                s.m_global_q[i] = (this->m_global_q[parent_id] * s.m_qua[i]).normalize();
//            }
//            else
//            {
//                s.m_global_q[i] = s.m_qua[i].normalize();
//            }
//        }
//
//        return s;
//    }
//
//    void RigidState::build()
//    {
//        if (!m_root)
//        {
//            return;
//        }
//
//        RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(m_root);
//
//        std::vector<std::shared_ptr<RigidBody2<DataType3f>>> all_nodes = root->getAllNode();
//        m_dq.resize(all_nodes.size());
//        m_r.resize(all_nodes.size());
//        m_qua.resize(all_nodes.size());
//        m_v.resize(all_nodes.size());
//
//        m_X.resize(all_nodes.size());
//        m_global_r.resize(all_nodes.size());
//        m_global_q.resize(all_nodes.size());
//
//        for (int i = 0; i < all_nodes.size(); ++i)
//        {
//            std::shared_ptr<RigidBody2<DataType3f>> cur_node = all_nodes[i];
//
//            m_dq[i] = cur_node->getParentJoint()->getdq();
//            m_r[i] = cur_node->getR();
//            m_qua[i] = cur_node->getQuaternion();
//            m_v[i] = cur_node->getV();
//
//            m_X[i] = cur_node->getX();
//            m_global_r[i] = cur_node->getGlobalR();
//            m_global_q[i] = cur_node->getGlobalQ();
//        }
//    }
//
//    RigidState RigidState::dydt(const RigidState& s0)
//    {
//        CTimer timer;
//        timer.start();
//
//        //const RigidState& s0 = static_cast<const RigidState&>(s0_);
//
//        //Vectornd<float> ddq;
//        //RigidBodyRoot<DataType3f>* root = s0.m_root;
//        RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(s0.m_root);
//
//
//        MatrixMN<float>& H = root->getH();
//        Vectornd<float>& C = root->getC();
//
//        CTimer timer_build_equation;
//        timer_build_equation.start();
//
//        // build motion equation
//        ForwardDynamicsSolver forward_dynamics;
//        forward_dynamics.setParent(root);
//        forward_dynamics.buildJointSpaceMotionEquation<RigidState>(s0, H, C, Vectornd<float>(root->getJointDof())); ///?????????????????????
//
//        timer_build_equation.stop();
//        std::cout << "  TIME * build equation: " << timer_build_equation.getElapsedTime() << std::endl;
//
//
//        CTimer timer_solve_equation;
//        timer_solve_equation.start();
//        Vectornd<float> ddq;
//        if (H.linearSolve(C.negative(), ddq))
//        {
//            timer_solve_equation.stop();
//            std::cout << "  TIME * solve equation: " << timer_solve_equation.getElapsedTime() << std::endl;
//
//
//            CTimer timer_dy;
//            timer_dy.start();
//
//
//            RigidState ds(root);
//            std::vector< std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>>& all_nodes = root->getAllParentidNodePair();
//
//            ds.m_dq.resize(all_nodes.size());
//            ds.m_r.resize(all_nodes.size());
//            ds.m_qua.resize(all_nodes.size());
//            ds.m_v.resize(all_nodes.size());
//
//            ds.m_X.resize(all_nodes.size());
//            ds.m_global_r.resize(all_nodes.size());
//            ds.m_global_q.resize(all_nodes.size());
//
//            // build index map
//            std::vector<int>& idx_map = root->getJointIdxMap();
//
//
//            for (int i = 0; i < all_nodes.size(); ++i)
//            {
//                int parent_id = all_nodes[i].first;
//                std::shared_ptr<RigidBody2<DataType3f>> cur_node = all_nodes[i].second;
//                std::shared_ptr<Joint> cur_joint = cur_node->getParentJoint();
//
//                int dof = cur_joint->getJointDOF();
//                if (dof > 0)
//                {
//                    MatrixMN<float> Xup = s0.m_X[i];
//                    MatrixMN<float> Xupinv = RigidUtil::transformationF2M(Xup.transpose());
//
//                    // d_dq
//                    Vectornd<float> d_dq(dof);
//                    for (int j = 0; j < dof; ++j)
//                    {
//                        d_dq(j) = ddq(idx_map[i] + j);
//                    }
//                    ds.m_dq[i] = d_dq;
//
//                    // d_vJ
//                    ds.m_v[i] = cur_joint->getS() * d_dq;// +RigidUtil::crm(cur_node->getV()) * (cur_joint->getS() * cur_joint->getdq());
//                                                         //ds.m_v[i] = cur_joint->getS() * d_dq + RigidUtil::crm(global_v[i]) * (cur_joint->getS() * cur_joint->getdq());
//
//                                                         // d_r, d_quaternion
//                    Vectornd<float> cur_v6 = s0.m_v[i];
//                    ds.m_qua[i] = s0.m_qua[i] * Quaternion<float>(cur_v6[0], cur_v6[1], cur_v6[2], 0) * 0.5;
//                    cur_v6 = Xupinv * cur_v6;
//                    Vectornd<float> cur_w(3);    cur_w[0] = cur_v6[0];    cur_w[1] = cur_v6[1];    cur_w[2] = cur_v6[2];
//                    Vectornd<float> d_r(3); d_r[0] = cur_v6[3]; d_r[1] = cur_v6[4]; d_r[2] = cur_v6[5];
//                    ds.m_r[i] = d_r + cur_w.cross(s0.m_r[i]);    // 3dim velocity: v = v0 + w x r;
//
//                }
//                else
//                {
//                    ds.m_dq[i] = Vectornd<float>();
//
//                    // d_vJ
//                    ds.m_v[i] = Vectornd<float>(6);                    // d_vJ, expressed in node frame
//
//                                                                            // d_r, d_quaternion
//                    ds.m_qua[i] = Quaternion<float>(0, 0, 0, 0);
//                    ds.m_r[i] = Vectornd<float>(3);
//                }
//            }
//
//            timer_dy.stop();
//            std::cout << "  TIME * ddq to dr&dQ: " << timer_dy.getElapsedTime() << std::endl;
//
//            timer.stop();
//            std::cout << "TIME * dydt:  " << timer.getElapsedTime() << std::endl;
//
//            return ds;
//
//        }
//        else
//        {
//            return RigidState();
//        }
//
//    }

}